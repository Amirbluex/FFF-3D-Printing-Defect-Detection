"""
Phase 7: Robustness Evaluation.

1. Generates corrupted versions of the test set images (annotations
   are unchanged -- corruption doesn't move object locations):
   - Lighting: darker, brighter
   - Noise: Gaussian, Salt & Pepper
   - Blur: Gaussian blur, Motion blur
   - Compression: JPEG quality 20%, JPEG quality 50%

2. Runs all three trained detectors against each corrupted set, using
   the SAME shared evaluation harness as Phase 6 (imported directly
   from evaluate.py) -- so degradation numbers are computed with
   identical methodology to the clean baseline, not a separate one.

3. Compares each corrupted result against the Phase 6 clean-test
   baseline (results/comparison_tables/full_comparison.json) to
   compute performance degradation per corruption type per model --
   this answers the brief's question: "which detector is more robust?"
"""

import json
import cv2
import numpy as np
from pathlib import Path

from evaluate import (
    PROJECT_ROOT, TEST_DIR,
    load_test_ground_truth, evaluate_predictions,
    run_yolo_family_inference, run_dino_inference,
)

CORRUPTED_DIR = PROJECT_ROOT / "data" / "corrupted"
ROBUSTNESS_DIR = PROJECT_ROOT / "results" / "robustness"

SEED = 42
rng = np.random.RandomState(SEED)


# Corruption functions
def darker(img, factor=0.4):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def brighter(img, factor=1.8):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def gaussian_noise(img, sigma=25):
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper(img, amount=0.02):
    out = img.copy()
    n_pixels = img.shape[0] * img.shape[1]

    n_salt = int(amount * n_pixels * 0.5)
    coords = [rng.randint(0, i, n_salt) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255

    n_pepper = int(amount * n_pixels * 0.5)
    coords = [rng.randint(0, i, n_pepper) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0

    return out


def _scaled_odd_ksize(img, fraction):
    width = img.shape[1]
    ksize = max(3, int(width * fraction))
    return ksize | 1  # force odd (required by these blur operations)


def gaussian_blur(img, fraction=0.015):
    ksize = _scaled_odd_ksize(img, fraction)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def motion_blur(img, fraction=0.02):
    ksize = _scaled_odd_ksize(img, fraction)
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(img, -1, kernel)


def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


CORRUPTIONS = {
    "lighting_darker": darker,
    "lighting_brighter": brighter,
    "noise_gaussian": gaussian_noise,
    "noise_salt_pepper": salt_and_pepper,
    "blur_gaussian": gaussian_blur,
    "blur_motion": motion_blur,
    "compression_jpeg20": lambda img: jpeg_compress(img, 20),
    "compression_jpeg50": lambda img: jpeg_compress(img, 50),
}


def generate_corrupted_sets():
    """
    Writes corrupted images to data/corrupted/<category>/images/.
    Annotations are NOT duplicated per category -- the original test
    set's _annotations.coco.json is reused directly during evaluation,
    since corruption doesn't change bounding box locations.
    """
    images_dir = TEST_DIR / "images"

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        out_dir = CORRUPTED_DIR / corruption_name / "images"

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"{corruption_name}: already generated, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in images_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            corrupted = corruption_fn(img)
            cv2.imwrite(str(out_dir / img_path.name), corrupted)
            count += 1

        print(f"{corruption_name}: {count} images generated -> {out_dir}")


def load_clean_baseline():
    with open(PROJECT_ROOT / "results" / "comparison_tables" / "full_comparison.json") as f:
        return json.load(f)


def compute_degradation(clean_metrics, corrupted_metrics, keys):
    """
    Absolute and relative (%) degradation for each key metric.
    Positive degradation = performance dropped under corruption.
    """
    degradation = {}
    for key in keys:
        clean_val = clean_metrics[key]
        corrupted_val = corrupted_metrics[key]
        abs_drop = clean_val - corrupted_val
        rel_drop_pct = (abs_drop / clean_val * 100) if clean_val > 0 else 0.0
        degradation[key] = {
            "clean": round(clean_val, 4),
            "corrupted": round(corrupted_val, 4),
            "absolute_drop": round(abs_drop, 4),
            "relative_drop_pct": round(rel_drop_pct, 2),
        }
    return degradation


DEGRADATION_METRICS = ["precision", "recall", "f1_score", "mAP50", "mAP50_95"]


def evaluate_model_on_corruption(model_key, model_path_or_checkpoint, model_type,
                                  images_by_id, anns_by_image, coco_id_to_contiguous,
                                  corruption_name):
    images_dir = CORRUPTED_DIR / corruption_name / "images"

    if model_type in ("yolov8", "rtdetr"):
        preds, targets, fps, mem = run_yolo_family_inference(
            model_path_or_checkpoint, images_by_id, anns_by_image,
            coco_id_to_contiguous, model_type, images_dir=images_dir
        )
    else:  # dino
        preds, targets, fps, mem = run_dino_inference(
            model_path_or_checkpoint, images_by_id, anns_by_image,
            coco_id_to_contiguous, images_dir=images_dir
        )

    return evaluate_predictions(preds, targets)


def main():
    print("=== Step 1: Generating corrupted test sets ===")
    generate_corrupted_sets()

    print("\n=== Step 2: Loading ground truth and clean baseline ===")
    images_by_id, anns_by_image, coco_id_to_contiguous, class_names = load_test_ground_truth()
    clean_baseline = load_clean_baseline()

    # Locate DINO's best checkpoint (same logic as evaluate.py)
    checkpoint_dirs = sorted((PROJECT_ROOT / "results" / "dino" / "run").glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[1]))
    with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
        dino_best_checkpoint = json.load(f)["best_model_checkpoint"]

    models = {
        "YOLOv8n": {
            "path": PROJECT_ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt",
            "type": "yolov8",
        },
        "RT-DETR-L": {
            "path": PROJECT_ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt",
            "type": "rtdetr",
        },
        "DEIMv2-Nano (DINO sub.)": {
            "path": dino_best_checkpoint,
            "type": "dino",
        },
    }

    print("\n=== Step 3: Evaluating each model against each corruption ===")
    full_results = {}

    for model_name, model_info in models.items():
        print(f"\n--- {model_name} ---")
        full_results[model_name] = {}

        for corruption_name in CORRUPTIONS.keys():
            print(f"  {corruption_name}...")
            corrupted_metrics = evaluate_model_on_corruption(
                model_name, model_info["path"], model_info["type"],
                images_by_id, anns_by_image, coco_id_to_contiguous,
                corruption_name,
            )

            clean_metrics = clean_baseline[model_name]
            degradation = compute_degradation(clean_metrics, corrupted_metrics, DEGRADATION_METRICS)

            full_results[model_name][corruption_name] = {
                "corrupted_metrics": corrupted_metrics,
                "degradation": degradation,
            }

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ROBUSTNESS_DIR / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n=== Done. Saved to {out_path} ===")

    print("\n=== Summary: mean mAP50 relative drop (%) across all corruptions, per model ===")
    for model_name in models:
        drops = [
            full_results[model_name][c]["degradation"]["mAP50"]["relative_drop_pct"]
            for c in CORRUPTIONS
        ]
        print(f"  {model_name}: {np.mean(drops):.2f}% average mAP50 drop "
              f"(lower = more robust)")


if __name__ == "__main__":
    main()