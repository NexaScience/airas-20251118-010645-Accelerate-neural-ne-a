"""src/preprocess.py
Data loading & preprocessing (ImageNet-1k via HuggingFace streaming).
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator, Tuple

import timm
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class HFImageStream(IterableDataset):
    """Stream-based image dataset to minimise RAM usage."""

    def __init__(self, hf_repo: str, split: str, transform, limit: int | None = None):
        super().__init__()
        self.ds = load_dataset(hf_repo, split=split, streaming=True, cache_dir=str(CACHE_DIR))
        self.tf = transform
        self.limit = limit

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        it = iter(self.ds)
        if self.limit is not None:
            it = itertools.islice(it, self.limit)
        for ex in it:
            img = ex["image"]
            lbl = int(ex["label"])
            yield self.tf(img), torch.tensor(lbl, dtype=torch.long)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    dummy = timm.create_model(cfg.model.name, pretrained=False, num_classes=cfg.model.num_classes)
    data_cfg = timm.data.resolve_model_data_config(dummy)
    tf_train = timm.data.create_transform(**data_cfg, is_training=True)
    tf_val = timm.data.create_transform(**data_cfg, is_training=False)
    limit = 2 if cfg.mode == "trial" else None
    train_ds = HFImageStream(cfg.dataset.hf_repo, "train", tf_train, limit)
    val_ds = HFImageStream(cfg.dataset.hf_repo, "validation", tf_val, limit)
    train_loader = DataLoader(train_ds, batch_size=cfg.dataset.batch.micro, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.dataset.batch.micro, num_workers=4, pin_memory=True)
    return train_loader, val_loader
