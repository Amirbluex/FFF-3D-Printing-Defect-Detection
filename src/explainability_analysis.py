"""
explainability_analysis.py

Phase 8: Explainability Analysis.

Selects one representative "good prediction", "bad prediction", and
"failure case" image per model (based on per-image IoU-matching
quality against ground truth), then generates an EigenCAM heatmap for
each, cropped tightly to the relevant ground-truth bounding box.

All three models use EigenCAM:
- YOLOv8n: targets the layer before the Detect head
- RT-DETR-L: targets the backbone, before the AIFI transformer stage
  (attention hook did not capture usable weights on this Ultralytics
  version -- documented substitution)
- DEIMv2-Nano (DINO sub.): targets the HGNetv2 backbone's final block
  (classic attention rollout confirmed architecturally inapplicable --
  deformable attention outputs sparse sampling points, not a dense
  attention matrix -- documented substitution, see
  attention_rollout.py docstring)

Usage:
    python src/explainability_analysis.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (
    PROJECT_ROOT, TEST_DIR,
    load_test_ground_truth, box_iou,
    run_yolo_family_inference, run_dino_inference,
)
from explainability.gradcam import generate_yolov8_heatmap
from explainability.attention_rollout import (
    generate_dino_heatmap_eigencam,
    generate_rtdetr_heatmap_eigencam_fallback,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "explainability"
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5
GOOD_IOU_THRESHOLD = 0.75
BAD_IOU_RANGE = (0.5, 0.65)


def classify_image_quality(pred, target):
    """
    Returns 'good', 'bad', 'failure', or None (no ground truth boxes
    to evaluate against) for a single image's predictions vs targets.
    """
    gt_boxes, gt_labels = target["boxes"], target["labels"]
    if len(gt_boxes) == 0:
        return None

    keep = pred["scores"] >= CONF_THRESHOLD
    pred_boxes = pred["boxes"][keep]
    pred_labels = pred["labels"][keep]
    pred_scores = pred["scores"][keep]

    order = np.argsort(-pred_scores)
    pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]

    gt_matched = [False] * len(gt_boxes)
    matched_ious = []
    n_false_positives = 0

    for pbox, plabel in zip(pred_boxes, pred_labels):
        best_iou, best_idx = 0.0, -1
        for i, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[i] or glabel != plabel:
                continue
            iou = box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= IOU_MATCH_THRESHOLD:
            gt_matched[best_idx] = True
            matched_ious.append(best_iou)
        else:
            n_false_positives += 1

    n_false_negatives = sum(1 for m in gt_matched if not m)

    if n_false_negatives > 0:
        return "failure"
    if not matched_ious:
        return None
    min_iou = min(matched_ious)
    if min_iou >= GOOD_IOU_THRESHOLD and n_false_positives == 0:
        return "good"
    if BAD_IOU_RANGE[0] <= min_iou < BAD_IOU_RANGE[1] or n_false_positives > 0:
        return "bad"
    return None


def select_representative_images(all_predictions, all_targets, images_by_id):
    """
    Returns (selected, selected_boxes):
      selected: {'good': image_id, 'bad': image_id, 'failure': image_id}
      selected_boxes: {'good': [x1,y1,x2,y2], ...} -- the first
        ground-truth box for that image, used to crop the EigenCAM
        input region.
    """
    image_ids = list(images_by_id.keys())
    selected = {"good": None, "bad": None, "failure": None}
    selected_boxes = {"good": None, "bad": None, "failure": None}

    for image_id, pred, target in zip(image_ids, all_predictions, all_targets):
        quality = classify_image_quality(pred, target)
        if quality and selected[quality] is None:
            selected[quality] = image_id
            selected_boxes[quality] = target["boxes"][0].tolist() if len(target["boxes"]) > 0 else None
        if all(v is not None for v in selected.values()):
            break

    return selected, selected_boxes


def save_heatmap_figure(original, overlay, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    images_by_id, anns_by_image, coco_id_to_contiguous, class_names = load_test_ground_truth()

    for category in ["good_predictions", "bad_predictions", "failure_cases"]:
        (RESULTS_DIR / category).mkdir(parents=True, exist_ok=True)

    # ---------- YOLOv8n ----------
    print("=== YOLOv8n ===")
    from ultralytics import YOLO
    yolo_model = YOLO(str(PROJECT_ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt"))

    preds, targets, _, _ = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "yolov8"
    )
    selected, selected_boxes = select_representative_images(preds, targets, images_by_id)
    print("Selected images:", selected)

    for quality, image_id in selected.items():
        if image_id is None:
            print(f"  No '{quality}' example found for YOLOv8n -- skipping")
            continue
        img_path = TEST_DIR / "images" / images_by_id[image_id]["file_name"]
        bbox = selected_boxes[quality]
        original, overlay, _ = generate_yolov8_heatmap(yolo_model, img_path, bbox=bbox)
        category_folder = f"{quality}_predictions" if quality != "failure" else "failure_cases"
        save_heatmap_figure(
            original, overlay, f"YOLOv8n EigenCAM ({quality})",
            RESULTS_DIR / category_folder / f"yolov8n_{quality}.png"
        )

    # ---------- RT-DETR-L ----------
    print("\n=== RT-DETR-L ===")
    from ultralytics import RTDETR
    rtdetr_model = RTDETR(str(PROJECT_ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt"))

    preds, targets, _, _ = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "rtdetr"
    )
    selected, selected_boxes = select_representative_images(preds, targets, images_by_id)
    print("Selected images:", selected)

    for quality, image_id in selected.items():
        if image_id is None:
            print(f"  No '{quality}' example found for RT-DETR-L -- skipping")
            continue
        img_path = TEST_DIR / "images" / images_by_id[image_id]["file_name"]
        bbox = selected_boxes[quality]
        original, overlay, _ = generate_rtdetr_heatmap_eigencam_fallback(rtdetr_model, img_path, bbox=bbox)
        category_folder = f"{quality}_predictions" if quality != "failure" else "failure_cases"
        save_heatmap_figure(
            original, overlay, f"RT-DETR-L EigenCAM* ({quality})",
            RESULTS_DIR / category_folder / f"rtdetr_{quality}.png"
        )

    # ---------- DEIMv2-Nano (DINO substitute) ----------
    print("\n=== DEIMv2-Nano (DINO sub.) ===")
    from transformers import AutoImageProcessor, Deimv2ForObjectDetection

    checkpoint_dirs = sorted((PROJECT_ROOT / "results" / "dino" / "run").glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[1]))
    with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
        best_checkpoint = json.load(f)["best_model_checkpoint"]

    dino_model = Deimv2ForObjectDetection.from_pretrained(best_checkpoint)
    dino_processor = AutoImageProcessor.from_pretrained(
        "harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers"
    )

    preds, targets, _, _ = run_dino_inference(
        best_checkpoint, images_by_id, anns_by_image, coco_id_to_contiguous
    )
    selected, selected_boxes = select_representative_images(preds, targets, images_by_id)
    print("Selected images:", selected)

    for quality, image_id in selected.items():
        if image_id is None:
            print(f"  No '{quality}' example found for DEIMv2-Nano -- skipping")
            continue
        img_path = TEST_DIR / "images" / images_by_id[image_id]["file_name"]
        bbox = selected_boxes[quality]
        original, overlay, _ = generate_dino_heatmap_eigencam(dino_model, dino_processor, img_path, bbox=bbox)
        category_folder = f"{quality}_predictions" if quality != "failure" else "failure_cases"
        save_heatmap_figure(
            original, overlay, f"DEIMv2-Nano EigenCAM* ({quality})",
            RESULTS_DIR / category_folder / f"dino_{quality}.png"
        )

    print(f"\nDone. Heatmaps saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()