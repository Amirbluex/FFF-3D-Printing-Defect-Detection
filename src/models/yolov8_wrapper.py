"""
Wraps Ultralytics YOLOv8 training/evaluation behind a common interface
so train.py can call any model (YOLOv8, RT-DETR, DINO) the same way.

Handles:
- Training with paper's hyperparameters, built-in augmentation disabled
  (since paper's grayscale/elastic/rotation augmentation was precomputed
  onto disk by augment_images.py)
- Automatic batch-size fallback on CUDA OOM
- Benchmarking (FPS, Parameters, GFLOPs, model size, mAP/precision/recall)
"""

import torch
from pathlib import Path
from ultralytics import YOLO

from ultralytics.utils.torch_utils import get_flops



class YOLOv8Wrapper:
    MODEL_VARIANT = "yolov8n.pt"  # paper's choice, Section 3.1

    def __init__(self, config, data_yaml_path, output_dir):
        self.config = config
        self.data_yaml_path = Path(data_yaml_path)
        self.output_dir = Path(output_dir)
        self.model = None
        self.attempt_log = []
        self.final_batch_size = None

    def train(self):
        batch_size = self.config["training"]["batch_size"]
        epochs = self.config["training"]["epochs"]
        imgsz = self.config["training"]["imgsz"]
        workers = self.config["training"]["workers"]
        seed = self.config["training"]["seed"]

        self.model = YOLO(self.MODEL_VARIANT)

        last_checkpoint = self.output_dir / "run" / "weights" / "last.pt"
        if last_checkpoint.exists():
            print(f"Found existing checkpoint at {last_checkpoint}, resuming...")
            self.model = YOLO(str(last_checkpoint))
            resume_training = True
        else:
            self.model = YOLO(self.MODEL_VARIANT)
            resume_training = False

        while batch_size >= 1:
            try:
                print(f"\nAttempting training with batch_size={batch_size}...")
                results = self.model.train(
                    data=str(self.data_yaml_path),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    optimizer="Adam",
                    workers=workers,
                    seed=seed,
                    patience=30,
                    project=str(self.output_dir),
                    name="run",
                    exist_ok=True,
                    resume=resume_training,


                    # Disable ALL built-in augmentation -- paper's
                    # grayscale + elastic + rotation was already baked
                    # into the training images on disk.
                    mosaic=0.0, mixup=0.0, copy_paste=0.0,
                    degrees=0.0, translate=0.0, scale=0.0,
                    shear=0.0, perspective=0.0,
                    flipud=0.0, fliplr=0.0,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    erasing=0.0, auto_augment=None,
                )
                self.attempt_log.append({"batch_size": batch_size, "status": "success"})
                self.final_batch_size = batch_size
                return results

            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"OOM at batch_size={batch_size}. Halving and retrying...")
                self.attempt_log.append({"batch_size": batch_size, "status": "OOM"})
                torch.cuda.empty_cache()
                batch_size = batch_size // 2

        raise RuntimeError("Training failed even at batch_size=1 -- hardware "
                            "insufficient for this configuration.")

    def benchmark(self):
        """
        Records FPS, Parameters, GFLOPs, model size, mAP/precision/recall
        -- required deliverables for Phase 3.
        """
        n_params = sum(p.numel() for p in self.model.model.parameters())
        gflops = get_flops(self.model.model, imgsz=self.config["training"]["imgsz"])

        val_results = self.model.val(data=str(self.data_yaml_path), split="test")
        speed_ms = val_results.speed
        total_ms_per_image = sum(v for k, v in speed_ms.items() if k != "loss")
        fps = 1000.0 / total_ms_per_image if total_ms_per_image > 0 else None

        weights_path = self.output_dir / "run" / "weights" / "best.pt"
        model_size_mb = (weights_path.stat().st_size / (1024 * 1024)
                          if weights_path.exists() else None)

        return {
            "parameters": n_params,
            "GFLOPs": round(gflops, 2),
            "FPS": round(fps, 2) if fps else None,
            "model_size_MB": round(model_size_mb, 2) if model_size_mb else None,
            "speed_breakdown_ms": speed_ms,
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        }

    def get_report_metadata(self):
        return {
            "model": "YOLOv8n",
            "paper_batch_size": self.config["training"]["batch_size"],
            "actual_batch_size_used": self.final_batch_size,
            "batch_size_reduced_due_to_hardware":
                self.final_batch_size != self.config["training"]["batch_size"],
            "attempt_log": self.attempt_log,
            "epochs": self.config["training"]["epochs"],
            "optimizer": "Adam",
            "imgsz": self.config["training"]["imgsz"],
        }