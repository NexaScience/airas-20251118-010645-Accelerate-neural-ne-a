"""src/train.py
Single experiment executor with full WandB logging, Optuna support, and the
carbon-aware C-FLAC controller implementation.
"""
from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import OmegaConf
from optuna.samplers import TPESampler
from optuna.trial import Trial
from torch import nn
from torch.utils.data import DataLoader

from src.model import build_model
from src.preprocess import build_dataloaders

# -----------------------------------------------------------------------------
# Global cache directory for ALL external files (datasets, checkpoints, …)
# -----------------------------------------------------------------------------
CACHE_DIR = Path(".cache").absolute()
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

################################################################################
# GPU power handling helpers ---------------------------------------------------
################################################################################
try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlInit,
    )

    nvmlInit()
    _NVML_AVAILABLE = True
    _NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover
    _NVML_AVAILABLE = False
    _NVML_HANDLE = None


def _power_usage_w() -> float:
    """Instantaneous GPU power draw in watts (0.0 if unavailable)."""
    if _NVML_AVAILABLE:
        return nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0  # mW → W
    return 0.0


def _set_power_cap(cap_watt: int) -> None:
    """Attempt to change the GPU power-limit; silently continue on failure."""
    try:
        subprocess.run(["nvidia-smi", "-pl", str(cap_watt)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

################################################################################
# Carbon-intensity fetcher (async HTTP) ----------------------------------------
################################################################################


class CarbonFetcher(Thread):
    """Background thread polling a carbon-intensity API (gCO₂ kWh⁻¹)."""

    def __init__(self, provider: str, polling_sec: int, stop_evt: Event, fallback: str):
        super().__init__(daemon=True)
        self.provider = provider.lower()
        self.polling_sec = polling_sec
        self._stop_evt = stop_evt
        self._latest = 400.0  # conservative default until first fetch
        self._fallback = fallback.lower()

    # ---------------------------------------------------------------------
    def latest(self) -> float:  # noqa: D401 – simple getter
        return float(self._latest)

    # ---------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 – thread loop
        import requests  # late import

        url_map = {
            "electricitymap": "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=FR",
            "watttime": "https://api.watttime.org/carbon",
        }
        while not self._stop_evt.is_set():
            try:
                url = url_map[self.provider]
                r = requests.get(url, timeout=5, headers={"Accept": "application/json"})
                if r.ok:
                    data = r.json()
                    self._latest = float(data.get("carbonIntensity", data.get("co2_g_per_kwh", 400)))
            except Exception:
                # fall back: keep previous value. If provider unreachable for long,
                # behaviour converges to E-FLAC (energy-only).
                pass
            finally:
                self._stop_evt.wait(self.polling_sec)

################################################################################
# Controllers ------------------------------------------------------------------
################################################################################


class BaseController:
    """Shared state + interface for all FLAC variants."""

    def __init__(self, cfg):
        t = cfg.training
        self.cfg = cfg
        self.eta = t.eta_min
        self.k = t.k
        self.momentum = t.initial_momentum
        self.alpha = 1.0  # surrogate curvature
        self.power_cap = getattr(t, "power_cap_fixed", 400)

    def update(self, loss: float, g2: float, carbon: float) -> Dict[str, float]:  # noqa: D401
        raise NotImplementedError


class EFlacController(BaseController):
    """Energy-only forward-looking adaptive controller (baseline)."""

    def update(self, loss: float, g2: float, carbon: float) -> Dict[str, float]:  # noqa: D401
        t = self.cfg.training
        self.eta = min(t.eta_max, 2 * loss / (self.alpha * g2 + 1e-12))
        rho = t.rho_smoothing
        k_star = math.sqrt(rho * self.eta * g2 / (2 * self.alpha * loss + 1e-12))
        self.k = int(max(1, min(t.k_max, round(k_star))))
        self.momentum = max(0.0, min(0.999, 1 - t.kappa * self.eta))
        self.alpha = rho * self.alpha + (1 - rho) * (2 * loss / (self.eta * g2 + 1e-12))
        return {
            "eta": self.eta,
            "k": self.k,
            "momentum": self.momentum,
            "power_cap": self.power_cap,
        }


class CFlacController(EFlacController):
    """Proposed carbon-aware controller extending E-FLAC with power-cap search."""

    def __init__(self, cfg, energy_model: Tuple[float, float]):
        super().__init__(cfg)
        self.a, self.b = energy_model  # E = k·(a + b·P)
        self.caps = list(cfg.training.power_caps)

    # ------------------------------------------------------------------
    def update(self, loss: float, g2: float, carbon: float) -> Dict[str, float]:  # noqa: D401
        stats = super().update(loss, g2, carbon)
        best_cap, best_cost = self.caps[0], float("inf")
        for p in self.caps:
            joule = self.k * (self.a + self.b * p)
            co2 = joule / 3.6e6 * carbon  # J → kWh (÷3.6e6) → kg (×g/1000)
            if co2 < best_cost:
                best_cap, best_cost = p, co2
        self.power_cap = best_cap
        _set_power_cap(best_cap)
        stats["power_cap"] = best_cap
        return stats

################################################################################
# Evaluation helper ------------------------------------------------------------
################################################################################


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_cls: int, max_batches: Optional[int]) -> Tuple[float, np.ndarray]:
    model.eval()
    total, correct = 0, 0
    preds, labels_all = [], []
    with torch.no_grad():
        for b, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            p = out.argmax(1)
            preds.append(p.cpu())
            labels_all.append(y.cpu())
            correct += (p == y).sum().item()
            total += y.size(0)
            if max_batches is not None and b + 1 >= max_batches:
                break
    preds = torch.cat(preds).numpy()
    labels_all = torch.cat(labels_all).numpy()
    cm = np.zeros((num_cls, num_cls), dtype=np.int64)
    for t, p in zip(labels_all, preds):
        cm[t, p] += 1
    return (correct / total if total else 0.0), cm

################################################################################
# Energy model fitting ---------------------------------------------------------
################################################################################


def _fit_energy_model(cfg) -> Tuple[float, float]:
    """Empirically fit a linear energy model E = a + b·P (per micro-batch)."""
    if not torch.cuda.is_available() or not _NVML_AVAILABLE:
        # Fallback heuristic based on typical A100 efficiency curve.
        return 30.0, 0.15

    device = torch.device("cuda")
    model = build_model(cfg).to(device).eval()
    dummy = torch.randn(cfg.dataset.batch.micro, 3, cfg.model.input_size, cfg.model.input_size, device=device)
    lbl = torch.zeros(cfg.dataset.batch.micro, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    energies = []
    for P in cfg.training.power_caps:
        _set_power_cap(P)
        torch.cuda.synchronize()
        time.sleep(0.2)  # settle clock
        joules = []
        for _ in range(cfg.training.power_cap_warmup_batches):
            t0 = time.time()
            power_w = _power_usage_w()
            out = model(dummy)
            loss = loss_fn(out, lbl)
            loss.backward()
            torch.cuda.synchronize()
            joules.append(power_w * (time.time() - t0))
            model.zero_grad(set_to_none=True)
        energies.append((P, float(np.mean(joules))))
    xs, ys = zip(*energies)
    b, a = np.polyfit(xs, ys, 1)  # y = a + b·x
    return float(a), float(b)

################################################################################
# Core training loop -----------------------------------------------------------
################################################################################


def _execute_training(cfg, hparams: Optional[Dict[str, Any]] = None, log_to_wandb: bool = True) -> float:
    if hparams:
        cfg = OmegaConf.merge(cfg, {"training": hparams})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.training.eta_min, momentum=cfg.training.initial_momentum, weight_decay=cfg.training.weight_decay)

    # ------------------------------------------------------------------
    if cfg.method.lower() in {"c-flac", "cflac", "proposed"}:
        a, b = _fit_energy_model(cfg)
        controller: BaseController = CFlacController(cfg, (a, b))
    else:
        controller = EFlacController(cfg)

    # Carbon signal -----------------------------------------------------
    stop_evt = Event()
    carbon_fetcher = CarbonFetcher(cfg.training.carbon_api.provider, cfg.training.carbon_api.polling_seconds, stop_evt, cfg.training.carbon_api.fallback)
    carbon_fetcher.start()

    # WandB setup -------------------------------------------------------
    wb = None
    if log_to_wandb and cfg.wandb.mode != "disabled":
        wb = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )

    gstep, best_val = 0, 0.0
    cum_kwh, cum_co2 = 0.0, 0.0

    try:
        for epoch in range(cfg.training.epochs):
            model.train()
            for b_idx, (x, y) in enumerate(train_loader):
                tic = time.time()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y) / controller.k
                loss.backward()

                if (b_idx + 1) % controller.k == 0:
                    g2 = 0.0
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                g2 += float((p.grad ** 2).sum())
                    stats = controller.update(loss.item() * controller.k, g2, carbon_fetcher.latest())
                    for pg in opt.param_groups:
                        pg["lr"] = stats["eta"]
                        pg["momentum"] = stats["momentum"]
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # Energy accounting --------------------------------
                step_energy_kwh = _power_usage_w() * (time.time() - tic) / 3600.0
                cum_kwh += step_energy_kwh
                cum_co2 += step_energy_kwh * carbon_fetcher.latest() / 1000.0

                if wb:
                    wb.log({
                        "train_loss": float(loss.item()),
                        "lr": controller.eta,
                        "k": controller.k,
                        "momentum": controller.momentum,
                        "power_cap": controller.power_cap,
                        "carbon_intensity": carbon_fetcher.latest(),
                        "step_energy_kwh": step_energy_kwh,
                        "cum_energy_kwh": cum_kwh,
                        "cum_co2_kg": cum_co2,
                        "epoch": epoch,
                    }, step=gstep)
                gstep += 1
                if cfg.mode == "trial" and gstep >= 3:
                    break
            # --------------------------------------------------------
            val_acc, cm = _evaluate(model, val_loader, device, cfg.model.num_classes, cfg.evaluation.val_max_batches)
            best_val = max(best_val, val_acc)
            if wb:
                wb.log({"val_acc": val_acc}, step=gstep)
            if cfg.mode == "trial":
                break
    finally:
        stop_evt.set()
        if wb:
            wb.summary["best_val_acc"] = best_val
            wb.summary["total_energy_kwh"] = cum_kwh
            wb.summary["total_co2_kg"] = cum_co2
            wb.summary["confusion_matrix"] = cm.tolist()
            wb.finish()
            print(f"[WandB] Run URL: {wb.url}")
    return cum_co2  # primary metric (minimise)

################################################################################
# Optuna wrapper & Hydra entry-point -------------------------------------------
################################################################################

def _sample_space(trial: Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, spec in space.items():
        t = str(spec["type"]).lower()
        if t == "uniform":
            out[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]))
        elif t == "loguniform":
            out[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "categorical":
            out[k] = trial.suggest_categorical(k, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search-space type: {t}")
    return out


@hydra.main(config_path="../config", config_name="config")
def _main(cfg) -> None:  # noqa: D401
    cfg_root = Path(__file__).resolve().parent.parent / "config"
    run_cfg = OmegaConf.load(cfg_root / "runs" / f"{cfg.run}.yaml")
    cfg = OmegaConf.merge(cfg, run_cfg)
    cfg.run_id = cfg.get("run_id", cfg.run)

    # Mode-specific overrides -----------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.evaluation.val_max_batches = 1
        cfg.dataset.batch.micro = min(cfg.dataset.batch.micro, 8)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    # Optuna ----------------------------------------------------------
    best_hp: Optional[Dict[str, Any]] = None
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0 and cfg.optuna.search_space:
        def _objective(trial: Trial) -> float:
            hp = _sample_space(trial, cfg.optuna.search_space)
            return _execute_training(cfg, hp, log_to_wandb=False)

        study = optuna.create_study(direction=cfg.optuna.direction, sampler=TPESampler(seed=42))
        study.optimize(_objective, n_trials=int(cfg.optuna.n_trials), show_progress_bar=True)
        best_hp = study.best_trial.params

    # Final training with best hyper-params (or default) --------------
    _execute_training(cfg, best_hp, log_to_wandb=True)


if __name__ == "__main__":
    _main()
