"""src/evaluate.py
Independent evaluation & visualisation pipeline executed AFTER all training
runs have finished.  Fixes previous JSON-serialisation issues by converting
NumPy / pandas objects to built-in Python types before writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

# -----------------------------------------------------------------------------
PRIMARY_METRIC = "total_co2_kg"  # must match train.py summary key
# -----------------------------------------------------------------------------

################################################################################
# Utilities --------------------------------------------------------------------
################################################################################

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help="JSON list of WandB run IDs")
    return p.parse_args()


# ------------------------------------------------------------------
# Safe JSON helper (handles NumPy / pandas objects)
# ------------------------------------------------------------------

def _json_default(o):  # noqa: D401 – custom encoder
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(path)


################################################################################
# Figure helpers ---------------------------------------------------------------
################################################################################

def _plot_learning_curve(rid: str, hist: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6, 4))
    if "train_loss" in hist:
        sns.lineplot(data=hist, x=hist.index, y="train_loss", ax=ax1, label="train_loss")
        ax1.set_ylabel("loss")
    if "val_acc" in hist:
        ax2 = ax1.twinx()
        sns.lineplot(data=hist, x=hist.index, y="val_acc", color="orange", ax=ax2, label="val_acc")
        ax2.set_ylabel("val_acc")
    plt.title(f"{rid} – learning curves")
    plt.tight_layout()
    fp = out / f"{rid}_learning_curve.pdf"
    fig.savefig(fp)
    plt.close(fig)
    print(fp)


def _plot_confusion_matrix(rid: str, cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{rid} – Confusion matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fp = out / f"{rid}_confusion_matrix.pdf"
    fig.savefig(fp)
    plt.close(fig)
    print(fp)

################################################################################
# Per-run export ---------------------------------------------------------------
################################################################################

def _export_single(run: "wandb.apis.public.Run", out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = run.history(pandas=True)
    summary = dict(run.summary._json_dict)
    cfg = dict(run.config)

    # Serialise metrics -------------------------------------------------
    _save_json(out_dir / "metrics.json", {"history": hist.to_dict(orient="list"), "summary": summary, "config": cfg})

    # Figures -----------------------------------------------------------
    _plot_learning_curve(run.id, hist, out_dir)
    if "confusion_matrix" in summary:
        _plot_confusion_matrix(run.id, np.asarray(summary["confusion_matrix"], dtype=int), out_dir)

    return {"primary": summary.get(PRIMARY_METRIC), "summary": summary}

################################################################################
# Aggregated analysis ----------------------------------------------------------
################################################################################

def _aggregate(all_info: Dict[str, Dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics_map: Dict[str, Dict[str, float]] = {}
    proposed_vals, baseline_vals = [], []
    best_prop = {"run_id": None, "value": float("inf")}
    best_base = {"run_id": None, "value": float("inf")}

    for rid, info in all_info.items():
        summ = info["summary"]
        for k, v in summ.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                metrics_map.setdefault(k, {})[rid] = float(v)
        # Categorise runs --------------------------------------------
        lower = rid.lower()
        if any(t in lower for t in {"proposed", "cflac", "c-flac"}):
            if summ.get(PRIMARY_METRIC) is not None:
                proposed_vals.append(float(summ[PRIMARY_METRIC]))
                if float(summ[PRIMARY_METRIC]) < best_prop["value"]:
                    best_prop = {"run_id": rid, "value": float(summ[PRIMARY_METRIC])}
        elif any(t in lower for t in {"baseline", "comparative", "e-flac", "eflac"}):
            if summ.get(PRIMARY_METRIC) is not None:
                baseline_vals.append(float(summ[PRIMARY_METRIC]))
                if float(summ[PRIMARY_METRIC]) < best_base["value"]:
                    best_base = {"run_id": rid, "value": float(summ[PRIMARY_METRIC])}

    gap = None
    if best_prop["run_id"] and best_base["run_id"]:
        gap = (best_base["value"] - best_prop["value"]) / best_base["value"] * 100.0

    # Statistical significance (Welch) ----------------------------------
    p_val = None
    if len(proposed_vals) > 1 and len(baseline_vals) > 1:
        _, p_val = stats.ttest_ind(proposed_vals, baseline_vals, equal_var=False, nan_policy="omit")
        p_val = float(p_val)

    _save_json(out / "aggregated_metrics.json", {
        "primary_metric": "Total CO₂ emissions (kg CO₂-e) to hit the target validation score; energy and time as secondary.",
        "metrics": metrics_map,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
        "p_value": p_val,
    })

    # Comparison figures ---------------------------------------------
    if PRIMARY_METRIC in metrics_map and metrics_map[PRIMARY_METRIC]:
        names, vals = zip(*metrics_map[PRIMARY_METRIC].items())
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(names), y=list(vals), ax=ax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylabel(PRIMARY_METRIC); plt.xticks(rotation=45, ha="right")
        plt.title("Across-run comparison (lower = better)")
        plt.tight_layout()
        fp1 = out / "comparison_bar_chart.pdf"
        fig.savefig(fp1); plt.close(fig); print(fp1)

        # Box plot by category
        cat = [
            ("Proposed" if "proposed" in rid.lower() or "cflac" in rid.lower() else "Baseline", v)
            for rid, v in metrics_map[PRIMARY_METRIC].items()
        ]
        df_box = pd.DataFrame(cat, columns=["Category", PRIMARY_METRIC])
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_box, x="Category", y=PRIMARY_METRIC, ax=ax2)
        plt.tight_layout()
        fp2 = out / "comparison_box_plot.pdf"
        fig2.savefig(fp2); plt.close(fig2); print(fp2)

################################################################################
# Entry-point ------------------------------------------------------------------
################################################################################

def main() -> None:  # noqa: D401
    args = _parse_args()
    res_dir = Path(args.results_dir)
    run_ids: List[str] = json.loads(args.run_ids)

    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    entity, project = cfg.wandb.entity, cfg.wandb.project

    api = wandb.Api()
    all_info: Dict[str, Dict[str, Any]] = {}
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        info = _export_single(run, res_dir / rid)
        all_info[rid] = info

    _aggregate(all_info, res_dir / "comparison")


if __name__ == "__main__":
    main()
