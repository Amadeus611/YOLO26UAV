# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models."""
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def _build_sample_weights(self, dataset):
        """Build per-image sampling weights for tail classes such as bus and truck."""
        if not getattr(self.args, "use_weighted_sampler", False):
            return None
        if getattr(self.args, "rect", False):
            LOGGER.warning("Weighted sampler is incompatible with rect=True training, disabling weighted sampling.")
            return None

        classes = np.concatenate([lb["cls"].flatten() for lb in dataset.labels], 0) if dataset.labels else np.array([])
        if classes.size == 0:
            return None

        nc = int(self.data["nc"])
        counts = np.bincount(classes.astype(int), minlength=nc).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_weights = np.power(1.0 / counts, float(getattr(self.args, "sample_weight_power", 1.0)))

        tail_classes = set(int(x) for x in getattr(self.args, "sample_tail_classes", []) if 0 <= int(x) < nc)
        tail_gain = float(getattr(self.args, "sample_tail_gain", 1.0))

        sample_weights = []
        for lb in dataset.labels:
            cls = lb["cls"].astype(int).flatten() if len(lb["cls"]) else np.array([], dtype=int)
            if cls.size == 0:
                sample_weights.append(1.0)
                continue
            weight = float(class_weights[cls].mean())
            if tail_classes and any(c in tail_classes for c in cls):
                weight *= tail_gain
            sample_weights.append(weight)
        return torch.as_tensor(sample_weights, dtype=torch.double)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        sampler = None
        if mode == "train" and rank == -1:
            weights = self._build_sample_weights(dataset)
            if weights is not None:
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                shuffle = False
                LOGGER.info("Using weighted sampler for tail classes during training")

        if sampler is not None:
            nd = torch.cuda.device_count()
            nw = min(os.cpu_count() // max(nd, 1), self.args.workers)
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205 + RANK)
            from ultralytics.data.build import InfiniteDataLoader, seed_worker

            effective_batch = min(batch_size, len(dataset))
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=effective_batch,
                shuffle=False,
                sampler=sampler,
                num_workers=nw,
                prefetch_factor=4 if nw > 0 else None,
                pin_memory=nd > 0,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=generator,
                drop_last=self.args.compile and mode == "train" and len(dataset) % effective_batch != 0,
            )

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def _auto_detect_tail_classes(self):
        """Auto-detect underrepresented classes and set tail_class_idx / sample_tail_classes."""
        nc = self.data["nc"]
        names = self.data.get("names", {})
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        if classes.size == 0:
            return
        counts = np.bincount(classes.astype(int), minlength=nc).astype(np.float32)
        median_freq = np.median(counts[counts > 0]) if np.any(counts > 0) else 1.0
        tail = [i for i in range(nc) if counts[i] < median_freq * 0.5 and counts[i] > 0]
        if not tail:
            return

        # Only update if the user left the lists empty (don't override explicit config)
        if not list(getattr(self.args, "tail_class_idx", [])):
            self.args.tail_class_idx = tail
            LOGGER.info(f"Auto-detected tail classes for loss boost: {tail} ({[names.get(i, i) for i in tail]})")
        if not list(getattr(self.args, "sample_tail_classes", [])):
            self.args.sample_tail_classes = tail
            LOGGER.info(f"Auto-detected tail classes for sampling: {tail} ({[names.get(i, i) for i in tail]})")

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance."""
        self._auto_detect_tail_classes()

        manual = list(getattr(self.args, "manual_class_weights", []))
        if manual:
            if len(manual) != self.data["nc"]:
                raise ValueError("manual_class_weights length must equal dataset nc")
            weights = np.asarray(manual, dtype=np.float32)
            weights = weights / weights.mean()
            self.model.class_weights = torch.from_numpy(weights).to(self.device)
            LOGGER.info(f"Manual class weights: {self.model.class_weights.cpu().numpy().round(3)}")
            return

        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw
        weights = weights / weights.mean()
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations."""
        plot_images(labels=batch, paths=batch["im_file"], fname=self.save_dir / f"train_batch{ni}.jpg", on_plot=self.on_plot)

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model."""
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        n = len(train_dataset)
        del train_dataset
        return super().auto_batch(max_num_obj, dataset_size=n)
