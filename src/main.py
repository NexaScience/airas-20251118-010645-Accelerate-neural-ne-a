"""src/main.py
Run-level orchestrator – merges configs & spawns src.train as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent.parent / "config"


@hydra.main(config_path="../config", config_name="config")
def main(cfg):  # noqa: D401
    run_file = CONFIG_DIR / "runs" / f"{cfg.run}.yaml"
    if not run_file.exists():
        raise FileNotFoundError(run_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_file))

    # Mode overrides ----------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.dataset.batch.micro = min(cfg.dataset.batch.micro, 8)
        cfg.evaluation.val_max_batches = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    print("#" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    cmd = [sys.executable, "-u", "-m", "src.train", f"run={cfg.run}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
