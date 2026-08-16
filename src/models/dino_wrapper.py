"""
Phase 5: DINO Detector Implementation.

IMPORTANT — DOCUMENTED SUBSTITUTION:
The official IDEA-Research/DINO repository requires compiling a custom
CUDA operator (ms_deform_attn_cuda) targeting Python 3.7/PyTorch 1.9/
CUDA 11.1. This was attempted and failed on this project's environment
(and separately, on a collaborator's machine). Rather than lose
substantial time to an unresolvable environment mismatch, this wrapper
uses DEIMv2 (DETR with Improved Matching v2) via HuggingFace
Transformers -- a modern, actively-maintained detector in the same
DETR/denoising-training lineage as DINO. Its architecture directly
implements DINO's core contributions:
  - Contrastive DeNoising training (config.num_denoising,
    label_noise_ratio, box_noise_scale) -- DINO's "denoising queries"
  - Iterative anchor box refinement (config.with_box_refine) -- DINO's
    "anchor refinement"
  - Deformable-attention transformer decoder (config.decoder_n_points)

This is a legitimate substitution for the assignment's learning
objectives, not an unrelated model. It must be disclosed explicitly
in the report's Methodology section.

Uses the Nano (HGNetv2 backbone) checkpoint, the smallest available
variant, chosen for the same reason yolov8n and rtdetr-l (smallest
available) were chosen in earlier phases: hardware constraints (6GB VRAM).

Unlike YOLOv8Wrapper/RTDETRWrapper (which wrap Ultralytics' simple
.train()/.val() API), this wrapper implements a full HuggingFace
Trainer-based training loop, since HF models don't share that API.
The same train()/benchmark()/get_report_metadata() interface is
preserved so train.py's dispatch logic requires no changes.
"""

import json
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoImageProcessor,
    Deimv2ForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class CocoDetectionDataset(Dataset):
    def __init__(self, split_dir: Path, image_processor, coco_id_to_contiguous):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.image_processor = image_processor
        self.coco_id_to_contiguous = coco_id_to_contiguous

        with open(self.split_dir / "_annotations.coco.json") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = coco["categories"]

        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.anns_by_image.get(image_id, [])
        coco_annotations = [
            {
                "image_id": image_id,
                "category_id": self.coco_id_to_contiguous[ann["category_id"]],  # remapped
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            }
            for ann in anns
        ]

        target = {"image_id": image_id, "annotations": coco_annotations}

        encoding = self.image_processor(
            images=image, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoded = {"pixel_values": torch.stack(pixel_values)}
    encoded["labels"] = [item["labels"] for item in batch]
    return encoded


class DINOWrapper:
    CHECKPOINT = "harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers"

    def __init__(self, config, data_yaml_path, output_dir):
        self.config = config
        self.data_processed_dir = Path(data_yaml_path).parent  # data/processed/
        self.output_dir = Path(output_dir)
        self.model = None
        self.image_processor = None
        self.trainer = None
        self.attempt_log = []
        self.final_batch_size = None
        self.overrides = self._load_overrides()
        self.class_names = self._load_class_names()

    def _load_overrides(self):
        override_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs" / "dino_config.yaml"
        )
        if override_path.exists():
            with open(override_path) as f:
                data = yaml.safe_load(f)
                return data.get("overrides", {}) if data else {}
        return {}

    def _load_class_names(self):
        with open(self.data_processed_dir / "train" / "_annotations.coco.json") as f:
            coco = json.load(f)
        cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])

        # Remap original COCO category_id (e.g. 1-5) to contiguous
        # 0-indexed labels (0-4), since the model's classification
        # head only has valid output indices [0, num_classes).
        self.coco_id_to_contiguous = {c["id"]: i for i, c in enumerate(cats_sorted)}
        return {i: c["name"] for i, c in enumerate(cats_sorted)}

    def train(self):
        epochs = self.config["training"]["epochs"]
        lr = self.overrides.get("learning_rate", 0.0001)
        patience = self.overrides.get("patience", 30)

        checkpoint_dir = self.output_dir / "run"
        existing_checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
        resume = len(existing_checkpoints) > 0

        if resume:
            print(f"Found existing checkpoint(s) at {checkpoint_dir}, "
                  f"most recent: {existing_checkpoints[-1].name} -- resuming training.")
        else:
            print("No existing checkpoint found -- starting training from scratch.")

        self.image_processor = AutoImageProcessor.from_pretrained(self.CHECKPOINT)

        train_dataset = CocoDetectionDataset(
            self.data_processed_dir / "train", self.image_processor, self.coco_id_to_contiguous
        )
        val_dataset = CocoDetectionDataset(
            self.data_processed_dir / "val", self.image_processor, self.coco_id_to_contiguous
        )

        id2label = self.class_names
        label2id = {v: k for k, v in id2label.items()}

        batch_size = 16  # starting point; Nano variant is much smaller than
                          # RT-DETR-L, but exact VRAM footprint is unverified

        while batch_size >= 1:
            try:
                print(f"\nAttempting training with batch_size={batch_size}...")

                self.model = Deimv2ForObjectDetection.from_pretrained(
                    self.CHECKPOINT,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,  # replacing the 80-class COCO head with our 5 classes
                )

                training_args = TrainingArguments(
                    output_dir=str(checkpoint_dir),
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    learning_rate=lr,
                    weight_decay=1e-4,
                    save_strategy="epoch",
                    eval_strategy="epoch",
                    logging_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    save_total_limit=2,
                    dataloader_num_workers=min(self.config["training"]["workers"], 2),
                    remove_unused_columns=False,
                    seed=self.config["training"]["seed"],
                )

                self.trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=collate_fn,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
                )

                start_time = time.time()
                self.trainer.train(resume_from_checkpoint=resume)
                self.training_time_hours = (time.time() - start_time) / 3600

                self.attempt_log.append({"batch_size": batch_size, "status": "success"})
                self.final_batch_size = batch_size
                return

            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"OOM at batch_size={batch_size}. Halving and retrying...")
                self.attempt_log.append({"batch_size": batch_size, "status": "OOM"})
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                resume = False  # a fresh batch size can't resume from a
                                 # checkpoint saved under a different batch size

        raise RuntimeError("Training failed even at batch_size=1 -- hardware "
                            "insufficient for this configuration.")

    def _compute_metrics(self, eval_pred):
        """
        mAP computation using torchmetrics, following the standard
        HuggingFace object detection fine-tuning recipe.
        """

        predictions, targets = eval_pred.predictions, eval_pred.label_ids

        metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")

        preds_formatted = []
        targets_formatted = []
        for pred, target in zip(predictions, targets):
            preds_formatted.append({
                "boxes": torch.tensor(pred["pred_boxes"]),
                "scores": torch.tensor(pred["scores"]) if "scores" in pred else torch.ones(len(pred["pred_boxes"])),
                "labels": torch.tensor(pred["pred_labels"]) if "pred_labels" in pred else torch.zeros(len(pred["pred_boxes"]), dtype=torch.long),
            })
            targets_formatted.append({
                "boxes": torch.tensor(target["boxes"]),
                "labels": torch.tensor(target["class_labels"]),
            })

        metric.update(preds_formatted, targets_formatted)
        result = metric.compute()

        return {
            "map": result["map"].item(),
            "map_50": result["map_50"].item(),
            "mar_100": result["mar_100"].item(),
        }

    def benchmark(self):
        """
        Records FPS, Parameters, model size, mAP/precision/recall
        -- required deliverables for Phase 5.

        Uses a manual inference loop with the documented
        post_process_object_detection() method, rather than relying
        on Trainer's internal prediction-gathering format (which
        doesn't match a simple per-image dict structure).
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        test_dataset = CocoDetectionDataset(
            self.data_processed_dir / "test", self.image_processor, self.coco_id_to_contiguous
        )

        device = next(self.model.parameters()).device
        self.model.eval()

        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

        inference_times = []

        with torch.no_grad():
            for idx in range(len(test_dataset)):
                image_id = test_dataset.image_ids[idx]
                img_info = test_dataset.images[image_id]
                img_path = test_dataset.images_dir / img_info["file_name"]
                image = Image.open(img_path).convert("RGB")

                inputs = self.image_processor(images=image, return_tensors="pt").to(device)

                start = time.time()
                outputs = self.model(**inputs)
                inference_times.append(time.time() - start)

                target_sizes = torch.tensor([image.size[::-1]])
                results = self.image_processor.post_process_object_detection(
                    outputs, threshold=0.0, target_sizes=target_sizes
                )[0]

                pred = {
                    "boxes": results["boxes"].cpu(),
                    "scores": results["scores"].cpu(),
                    "labels": results["labels"].cpu(),
                }

                gt_anns = test_dataset.anns_by_image.get(image_id, [])
                gt_boxes = []
                gt_labels = []
                for ann in gt_anns:
                    x, y, w, h = ann["bbox"]
                    gt_boxes.append([x, y, x + w, y + h])  # xywh -> xyxy
                    gt_labels.append(self.coco_id_to_contiguous[ann["category_id"]])

                target = {
                    "boxes": torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0, 4)),
                    "labels": torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long),
                }

                metric.update([pred], [target])

        result = metric.compute()

        n_params = sum(p.numel() for p in self.model.parameters())
        avg_inference_s = sum(inference_times) / len(inference_times)
        fps = 1.0 / avg_inference_s

        checkpoint_dirs = sorted((self.output_dir / "run").glob("checkpoint-*"))
        model_size_mb = None
        if checkpoint_dirs:
            safetensors = checkpoint_dirs[-1] / "model.safetensors"
            if safetensors.exists():
                model_size_mb = safetensors.stat().st_size / (1024 * 1024)

        return {
            "parameters": n_params,
            "FPS": round(fps, 2),
            "model_size_MB": round(model_size_mb, 2) if model_size_mb else None,
            "mAP50": result["map_50"].item(),
            "mAP50_95": result["map"].item(),
            "recall": result["mar_100"].item(),
        }

    def get_report_metadata(self):
        return {
            "model": "DEIMv2-Nano (documented substitution for DINO)",
            "substitution_reason": self.overrides.get(
                "substitution_reason",
                "Official DINO requires CUDA compilation incompatible with this environment"
            ),
            "actual_batch_size_used": self.final_batch_size,
            "attempt_log": self.attempt_log,
            "epochs": self.config["training"]["epochs"],
            "learning_rate": self.overrides.get("learning_rate", 0.0001),
            "training_time_hours": getattr(self, "training_time_hours", None),
        }