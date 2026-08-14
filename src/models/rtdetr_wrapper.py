"""
Wraps Ultralytics RT-DETR training/evaluation behind the same common
interface as YOLOv8Wrapper, so train.py can call it identically.
Using Ultralytics' RT-DETR implementation (not the standalone
HuggingFace port) specifically so training conditions -- optimizer,
augmentation handling, logging, data format -- stay consistent with
the YOLOv8 baseline, per the project's "identical conditions" 
requirement.

Handles:
- Training with paper's hyperparameters, built-in augmentation disabled
  (paper's grayscale/elastic/rotation augmentation was precomputed onto
  disk by augment_images.py -- same as YOLOv8)
- Automatic batch-size fallback on CUDA OOM (RT-DETR's transformer
  decoder is more memory-hungry than YOLOv8n, so this matters more here)
- Resume support across interrupted sessions
- Benchmarking (FPS, Parameters, GFLOPs, model size, mAP/precision/recall)
"""

import torch
from pathlib import Path
from ultralytics import RTDETR


class RTDETRWrapper:
    MODEL_VARIANT = "rtdetr-l.pt"  # smallest RT-DETR variant Ultralytics ships

    def __init__(self, config, data_yaml_path, output_dir):
        self.config = config
        self.data_yaml_path = Path(data_yaml_path)
        self.output_dir = Path(output_dir)
        self.model = None
        self.attempt_log = []
        self.final_batch_size = None
        self.overrides = self._load_overrides()

    def _load_overrides(self):
        import yaml
        override_path = Path(__file__).resolve().parent.parent.parent / "configs" / "rtdetr_config.yaml"
        if override_path.exists():
            with open(override_path) as f:
                data = yaml.safe_load(f)
                return data.get("overrides", {}) if data else {}
        return {}

    def train(self):
        # RT-DETR-L (~33M params) is far more memory-hungry than
        # YOLOv8n (~3M params). Starting at the paper's batch=64 wastes
        # several failed spin-up/teardown cycles we already know will
        # OOM. Cap the starting point lower for this model specifically.
        batch_size = min(self.config["training"]["batch_size"], 8)
        epochs = self.config["training"]["epochs"]
        imgsz = self.config["training"]["imgsz"]
        # Reduced from shared config value -- RT-DETR's larger memory
        # footprint combined with 12GB system RAM makes fewer parallel
        # dataloader workers safer here specifically.
        workers = min(self.config["training"]["workers"], 2)
        seed = self.config["training"]["seed"]

        last_checkpoint = self.output_dir / "run" / "weights" / "last.pt"
        if last_checkpoint.exists():
            print(f"Found existing checkpoint at {last_checkpoint}, resuming...")
            self.model = RTDETR(str(last_checkpoint))
            resume_training = True
        else:
            self.model = RTDETR(self.MODEL_VARIANT)
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
                    lr0=self.overrides.get("lr0", 0.01),

                    # Disable ALL built-in augmentation -- same reasoning
                    # as YOLOv8: paper's grayscale + elastic + rotation
                    # was already baked into the training images on disk.
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
                # A fresh OOM retry should not also carry a stale
                # "resume" flag pointed at a checkpoint from a
                # different batch size's partial run
                resume_training = False

        raise RuntimeError("Training failed even at batch_size=1 -- hardware "
                            "insufficient for this configuration.")

    def benchmark(self):
        """
        Records FPS, Parameters, GFLOPs, model size, mAP/precision/recall
        -- required deliverables for Phase 4.
        """
        from ultralytics.utils.torch_utils import get_flops

        n_params = sum(p.numel() for p in self.model.model.parameters())
        gflops = get_flops(self.model.model, imgsz=self.config["training"]["imgsz"])

        val_results = self.model.val(data=str(self.data_yaml_path), split="test", workers=0)
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
            "model": "RT-DETR-L",
            "paper_batch_size": self.config["training"]["batch_size"],
            "actual_batch_size_used": self.final_batch_size,
            "batch_size_reduced_due_to_hardware":
                self.final_batch_size != self.config["training"]["batch_size"],
            "attempt_log": self.attempt_log,
            "epochs": self.config["training"]["epochs"],
            "optimizer": "Adam",
            "imgsz": self.config["training"]["imgsz"],
        }