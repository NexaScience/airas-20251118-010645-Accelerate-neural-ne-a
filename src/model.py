"""src/model.py
Model construction utilities – honours .cache/ for all checkpoints.
"""
from __future__ import annotations

import os
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

CACHE_DIR = Path(".cache").absolute()
CACHE_DIR.mkdir(exist_ok=True, parents=True)
# Ensure timm/torch hub also respect .cache/
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))


def _download_hf(repo: str) -> str:
    """Download common weight filenames from HF repo into .cache/."""
    for fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]:
        try:
            return hf_hub_download(repo_id=repo, filename=fname, cache_dir=str(CACHE_DIR))
        except Exception:
            continue
    raise FileNotFoundError(f"No weight file found for {repo}")


def build_model(cfg: DictConfig):
    """Return a timm model instance as defined in cfg."""
    model = timm.create_model(cfg.model.name, pretrained=False, num_classes=cfg.model.num_classes)
    if cfg.model.get("hf_repo"):
        weights = _download_hf(cfg.model.hf_repo)
        state = torch.load(weights, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    elif bool(cfg.model.get("pretrained", True)):
        # timm will now download into TORCH_HOME (.cache/)
        model = timm.create_model(cfg.model.name, pretrained=True, num_classes=cfg.model.num_classes)
    return model
