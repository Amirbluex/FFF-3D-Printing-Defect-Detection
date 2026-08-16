"""
evaluate.py

Phase 6: Quantitative Performance Comparison.

Runs a SHARED evaluation harness against all three trained detectors
(YOLOv8n, RT-DETR-L, DEIMv2-Nano), computing metrics the same way for
each model -- unlike the individual performance_report.json files,
which used each framework's own (sometimes differently-defined)
built-in validators. This produces genuinely comparable numbers.

Detection metrics (via greedy IoU>=0.5 matching at conf=0.25, plus
torchmetrics for mAP): Precision, Recall, F1, mean IoU (of true
positive matches), Dice, mAP50, mAP50-95.

Computational metrics: FPS, Inference time (ms), Parameters, GFLOPs
(where available), GPU memory (peak, measured freshly here for
consistency), Model size, Training time (where meaningfully measurable).

Usage:
    python src/evaluate.py
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor, Deimv2ForObjectDetection
from ultralytics import YOLO, RTDETR



PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
COMPARISON_DIR = PROJECT_ROOT / "results" / "comparison_tables"
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5


def load_test_ground_truth():
    with open(TEST_DIR / "_annotations.coco.json") as f:
        coco = json.load(f)
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    coco_id_to_contiguous = {c["id"]: i for i, c in enumerate(cats_sorted)}
    class_names = [c["name"] for c in cats_sorted]

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    return images_by_id, anns_by_image, coco_id_to_contiguous, class_names


def box_iou(box1, box2):
    """xyxy format IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                  conf_threshold, iou_threshold):
    """
    Greedy matching (highest-confidence predictions matched first) to
    compute TP/FP/FN and mean IoU of matched pairs, same convention
    used across standard detection evaluation.
    """
    keep = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    order = np.argsort(-pred_scores)
    pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]

    gt_matched = [False] * len(gt_boxes)
    tp, fp = 0, 0
    matched_ious = []

    for pbox, plabel in zip(pred_boxes, pred_labels):
        best_iou, best_idx = 0.0, -1
        for i, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[i] or glabel != plabel:
                continue
            iou = box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_idx] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, matched_ious


def evaluate_predictions(all_predictions, all_targets):
    """
    all_predictions / all_targets: list of dicts with keys
    boxes (xyxy), labels, scores (predictions only) -- one dict per image.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []

    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for pred, target in zip(all_predictions, all_targets):
        tp, fp, fn, ious = greedy_match(
            pred["boxes"], pred["labels"], pred["scores"],
            target["boxes"], target["labels"],
            CONF_THRESHOLD, IOU_MATCH_THRESHOLD,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        map_metric.update(
            [{"boxes": torch.tensor(pred["boxes"]), "scores": torch.tensor(pred["scores"]),
              "labels": torch.tensor(pred["labels"])}],
            [{"boxes": torch.tensor(target["boxes"]), "labels": torch.tensor(target["labels"])}],
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    # Dice coefficient is a monotonic transform of IoU: Dice = 2*IoU / (1 + IoU)
    mean_dice = 2 * mean_iou / (1 + mean_iou) if mean_iou > 0 else 0.0

    map_result = map_metric.compute()

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "mAP50": round(map_result["map_50"].item(), 4),
        "mAP50_95": round(map_result["map"].item(), 4),
    }


def run_yolo_family_inference(model_path, images_by_id, anns_by_image, coco_id_to_contiguous, model_type):

    ModelClass = YOLO if model_type == "yolov8" else RTDETR
    model = ModelClass(str(model_path))

    all_predictions, all_targets = [], []
    inference_times = []

    torch.cuda.reset_peak_memory_stats()

    for image_id, img_info in images_by_id.items():
        img_path = TEST_DIR / "images" / img_info["file_name"]

        start = time.time()
        results = model.predict(str(img_path), verbose=False, conf=0.001)[0]
        inference_times.append(time.time() - start)

        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_scores = results.boxes.conf.cpu().numpy()
        pred_labels = results.boxes.cls.cpu().numpy().astype(int)

        gt_anns = anns_by_image.get(image_id, [])
        gt_boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt_anns]
        gt_labels = [coco_id_to_contiguous[a["category_id"]] for a in gt_anns]

        all_predictions.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})
        all_targets.append({"boxes": np.array(gt_boxes) if gt_boxes else np.zeros((0, 4)),
                             "labels": np.array(gt_labels) if gt_labels else np.zeros((0,))})

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fps = 1.0 / (sum(inference_times) / len(inference_times))

    return all_predictions, all_targets, fps, peak_memory_mb


def run_dino_inference(checkpoint_path, images_by_id, anns_by_image, coco_id_to_contiguous):

    model = Deimv2ForObjectDetection.from_pretrained(str(checkpoint_path))
    image_processor = AutoImageProcessor.from_pretrained(
        "harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    all_predictions, all_targets = [], []
    inference_times = []

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for image_id, img_info in images_by_id.items():
            img_path = TEST_DIR / "images" / img_info["file_name"]
            image = Image.open(img_path).convert("RGB")

            inputs = image_processor(images=image, return_tensors="pt").to(device)

            start = time.time()
            outputs = model(**inputs)
            inference_times.append(time.time() - start)

            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.001, target_sizes=target_sizes
            )[0]

            pred_boxes = results["boxes"].cpu().numpy()
            pred_scores = results["scores"].cpu().numpy()
            pred_labels = results["labels"].cpu().numpy()

            gt_anns = anns_by_image.get(image_id, [])
            gt_boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt_anns]
            gt_labels = [coco_id_to_contiguous[a["category_id"]] for a in gt_anns]

            all_predictions.append({"boxes": pred_boxes, "labels": pred_labels, "scores": pred_scores})
            all_targets.append({"boxes": np.array(gt_boxes) if gt_boxes else np.zeros((0, 4)),
                                 "labels": np.array(gt_labels) if gt_labels else np.zeros((0,))})

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    fps = 1.0 / (sum(inference_times) / len(inference_times))

    return all_predictions, all_targets, fps, peak_memory_mb


def load_computational_metadata():
    with open(PROJECT_ROOT / "results" / "baseline_yolov8" / "performance_report.json") as f:
        yolov8_meta = json.load(f)
    with open(PROJECT_ROOT / "results" / "rtdetr" / "performance_report.json") as f:
        rtdetr_meta = json.load(f)
    with open(PROJECT_ROOT / "results" / "dino" / "performance_report.json") as f:
        dino_meta = json.load(f)
    return yolov8_meta, rtdetr_meta, dino_meta


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    images_by_id, anns_by_image, coco_id_to_contiguous, class_names = load_test_ground_truth()
    yolov8_meta, rtdetr_meta, dino_meta = load_computational_metadata()

    results = {}

    print("Evaluating YOLOv8n...")
    preds, targets, fps, mem = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "yolov8"
    )
    results["YOLOv8n"] = evaluate_predictions(preds, targets)
    results["YOLOv8n"].update({
        "FPS": round(fps, 2), "GPU_memory_MB": round(mem, 2),
        "parameters": yolov8_meta["parameters"], "GFLOPs": yolov8_meta["GFLOPs"],
        "model_size_MB": yolov8_meta["model_size_MB"],
        "training_time_hours": yolov8_meta.get("training_time_hours"),
    })
    results["YOLOv8n"]["inference_time_ms"] = round(1000 / results["YOLOv8n"]["FPS"], 2)

    print("Evaluating RT-DETR-L...")
    preds, targets, fps, mem = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "rtdetr"
    )
    results["RT-DETR-L"] = evaluate_predictions(preds, targets)
    results["RT-DETR-L"].update({
        "FPS": round(fps, 2), "GPU_memory_MB": round(mem, 2),
        "parameters": rtdetr_meta["parameters"], "GFLOPs": rtdetr_meta["GFLOPs"],
        "model_size_MB": rtdetr_meta["model_size_MB"],
        "training_time_hours": rtdetr_meta.get("training_time_hours"),  # cumulative
                                       # across multiple resumed sessions -- see
                                       # performance_report.json's training_time_note
    })
    results["RT-DETR-L"]["inference_time_ms"] = round(1000 / results["RT-DETR-L"]["FPS"], 2)

    print("Evaluating DEIMv2-Nano (DINO substitute)...")
    checkpoint_dirs = sorted((PROJECT_ROOT / "results" / "dino" / "run").glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[1]))
    with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
        best_checkpoint = json.load(f)["best_model_checkpoint"]

    preds, targets, fps, mem = run_dino_inference(
        best_checkpoint, images_by_id, anns_by_image, coco_id_to_contiguous
    )
    results["DEIMv2-Nano (DINO sub.)"] = evaluate_predictions(preds, targets)
    results["DEIMv2-Nano (DINO sub.)"].update({
        "FPS": round(fps, 2), "GPU_memory_MB": round(mem, 2),
        "parameters": dino_meta["parameters"], "GFLOPs": None,  # not computed:
                                                                   # no integrated FLOP
                                                                   # counter for HF
                                                                   # transformer models
                                                                   # in this pipeline
        "model_size_MB": dino_meta["model_size_MB"],
        "training_time_hours": dino_meta.get("training_time_hours"),
    })
    results["DEIMv2-Nano (DINO sub.)"]["inference_time_ms"] = round(
        1000 / results["DEIMv2-Nano (DINO sub.)"]["FPS"], 2
    )

    out_path = COMPARISON_DIR / "full_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved unified comparison to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()