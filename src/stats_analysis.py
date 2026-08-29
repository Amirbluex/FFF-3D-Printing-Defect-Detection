"""
Phase 9: Statistical Analysis.

Goes beyond simple point-estimate comparison (Phase 6) with proper
paired statistical tests, since all three models are evaluated on the
SAME 195 test images / same ground-truth boxes:

- Bootstrap confidence intervals: resample images with replacement,
  recompute aggregate metrics each time -> percentile 95% CI.
- Paired permutation test: per-image F1 scores, paired by image,
  tests whether one model's mean F1 is significantly different from
  another's.
- Wilcoxon signed-rank test: same per-image F1 pairs, non-parametric
  alternative (doesn't assume normally-distributed differences).
- McNemar's test: per-ground-truth-box binary correct/incorrect
  outcome, paired by the SAME box across two models -- directly
  answers "do these two models disagree on which specific defects
  they catch," per the brief's own phrasing ("matched detections").
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from evaluate import (
    PROJECT_ROOT, TEST_DIR,
    load_test_ground_truth, evaluate_predictions, box_iou,
    run_yolo_family_inference, run_dino_inference,
)

STATS_DIR = PROJECT_ROOT / "results" / "statistical_tests"
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5
N_BOOTSTRAP = 1000
SEED = 42

rng = np.random.RandomState(SEED)


def per_image_f1(pred, target):
    """
    F1 score for a single image's predictions vs ground truth,
    using the same greedy IoU matching convention as evaluate.py.
    """
    gt_boxes, gt_labels = target["boxes"], target["labels"]
    keep = pred["scores"] >= CONF_THRESHOLD
    pred_boxes = pred["boxes"][keep]
    pred_labels = pred["labels"][keep]
    pred_scores = pred["scores"][keep]

    order = np.argsort(-pred_scores)
    pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]

    gt_matched = [False] * len(gt_boxes)
    tp, fp = 0, 0

    for pbox, plabel in zip(pred_boxes, pred_labels):
        best_iou, best_idx = 0.0, -1
        for i, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[i] or glabel != plabel:
                continue
            iou = box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_MATCH_THRESHOLD:
            tp += 1
            gt_matched[best_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def per_gt_box_outcomes(pred, target):
    """
    Returns a list of booleans, one per ground-truth box in this
    image: True if correctly detected (matched at IoU>=0.5, correct
    class), False if missed. Used for McNemar's test, which needs
    paired binary outcomes on the SAME items across two models.
    """
    gt_boxes, gt_labels = target["boxes"], target["labels"]
    keep = pred["scores"] >= CONF_THRESHOLD
    pred_boxes = pred["boxes"][keep]
    pred_labels = pred["labels"][keep]
    pred_scores = pred["scores"][keep]

    order = np.argsort(-pred_scores)
    pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]

    gt_matched = [False] * len(gt_boxes)

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

    return gt_matched


def bootstrap_ci(all_predictions, all_targets, metric_key, n_bootstrap=N_BOOTSTRAP):
    """
    Resamples images (with replacement) n_bootstrap times, recomputes
    the aggregate metric each time, returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(all_predictions)
    point_estimate = evaluate_predictions(all_predictions, all_targets)[metric_key]

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        resampled_preds = [all_predictions[i] for i in idx]
        resampled_targets = [all_targets[i] for i in idx]
        score = evaluate_predictions(resampled_preds, resampled_targets)[metric_key]
        bootstrap_scores.append(score)

    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    return point_estimate, ci_lower, ci_upper


def paired_permutation_test(scores_a, scores_b, n_permutations=10000):
    """
    Tests whether the mean difference between paired per-image scores
    (scores_a - scores_b) is significantly different from zero, by
    randomly flipping the sign of each pair's difference many times
    to build a null distribution.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diffs = scores_a - scores_b
    observed_mean_diff = diffs.mean()

    permuted_diffs = []
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        permuted_diffs.append((diffs * signs).mean())
    permuted_diffs = np.array(permuted_diffs)

    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_mean_diff))

    # Effect size: Cohen's d for paired samples
    effect_size = observed_mean_diff / diffs.std() if diffs.std() > 0 else 0.0

    return {
        "observed_mean_diff": round(float(observed_mean_diff), 4),
        "p_value": round(float(p_value), 4),
        "effect_size_cohens_d": round(float(effect_size), 4),
    }


def wilcoxon_test(scores_a, scores_b):
    diffs = np.array(scores_a) - np.array(scores_b)
    if np.all(diffs == 0):
        return {"statistic": None, "p_value": 1.0, "note": "All paired differences are zero"}
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": round(float(statistic), 4), "p_value": round(float(p_value), 4)}


def mcnemar_test(outcomes_a, outcomes_b):
    """
    outcomes_a/b: lists of booleans, same length, paired by the same
    ground-truth box.
    """
    both_correct = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a and b)
    a_only = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a and not b)
    b_only = sum(1 for a, b in zip(outcomes_a, outcomes_b) if not a and b)
    both_wrong = sum(1 for a, b in zip(outcomes_a, outcomes_b) if not a and not b)

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=(a_only + b_only < 25), correction=True)

    return {
        "contingency_table": {
            "both_correct": both_correct, "a_only_correct": a_only,
            "b_only_correct": b_only, "both_wrong": both_wrong,
        },
        "statistic": round(float(result.statistic), 4),
        "p_value": round(float(result.pvalue), 4),
    }


def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    images_by_id, anns_by_image, coco_id_to_contiguous, class_names = load_test_ground_truth()

    print("Running inference for all three models on clean test set...")

    print("  YOLOv8n...")
    yolo_preds, yolo_targets, _, _ = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "yolov8"
    )

    print("  RT-DETR-L...")
    rtdetr_preds, rtdetr_targets, _, _ = run_yolo_family_inference(
        PROJECT_ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt",
        images_by_id, anns_by_image, coco_id_to_contiguous, "rtdetr"
    )

    print("  DEIMv2-Nano (DINO sub.)...")
    checkpoint_dirs = sorted((PROJECT_ROOT / "results" / "dino" / "run").glob("checkpoint-*"),
                              key=lambda p: int(p.name.split("-")[1]))
    with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
        best_checkpoint = json.load(f)["best_model_checkpoint"]
    dino_preds, dino_targets, _, _ = run_dino_inference(
        best_checkpoint, images_by_id, anns_by_image, coco_id_to_contiguous
    )

    models = {
        "YOLOv8n": (yolo_preds, yolo_targets),
        "RT-DETR-L": (rtdetr_preds, rtdetr_targets),
        "DEIMv2-Nano (DINO sub.)": (dino_preds, dino_targets),
    }

    results = {"bootstrap_ci": {}, "paired_permutation": {}, "wilcoxon": {}, "mcnemar": {}}

    # ---------- Bootstrap CIs ----------
    print("\n=== Bootstrap Confidence Intervals (1000 resamples) ===")
    for model_name, (preds, targets) in models.items():
        results["bootstrap_ci"][model_name] = {}
        for metric_key in ["precision", "recall", "f1_score", "mAP50", "mAP50_95"]:
            point, lo, hi = bootstrap_ci(preds, targets, metric_key)
            results["bootstrap_ci"][model_name][metric_key] = {
                "point_estimate": round(point, 4), "ci_95_lower": round(lo, 4), "ci_95_upper": round(hi, 4)
            }
            print(f"  {model_name} {metric_key}: {point:.4f} [{lo:.4f}, {hi:.4f}]")

    # ---------- Per-image F1 for paired tests ----------
    per_image_f1_scores = {}
    for model_name, (preds, targets) in models.items():
        per_image_f1_scores[model_name] = [per_image_f1(p, t) for p, t in zip(preds, targets)]

    # Per-GT-box outcomes for McNemar's 
    per_gt_outcomes = {}
    for model_name, (preds, targets) in models.items():
        all_outcomes = []
        for p, t in zip(preds, targets):
            all_outcomes.extend(per_gt_box_outcomes(p, t))
        per_gt_outcomes[model_name] = all_outcomes

    # Pairwise tests 
    model_names = list(models.keys())
    pairs = [(model_names[i], model_names[j]) for i in range(len(model_names)) for j in range(i + 1, len(model_names))]

    print("\n=== Paired Permutation Tests (per-image F1) ===")
    for a, b in pairs:
        key = f"{a} vs {b}"
        result = paired_permutation_test(per_image_f1_scores[a], per_image_f1_scores[b])
        results["paired_permutation"][key] = result
        print(f"  {key}: mean diff={result['observed_mean_diff']}, p={result['p_value']}, "
              f"Cohen's d={result['effect_size_cohens_d']}")

    print("\n=== Wilcoxon Signed-Rank Tests (per-image F1) ===")
    for a, b in pairs:
        key = f"{a} vs {b}"
        result = wilcoxon_test(per_image_f1_scores[a], per_image_f1_scores[b])
        results["wilcoxon"][key] = result
        print(f"  {key}: statistic={result['statistic']}, p={result['p_value']}")

    print("\n=== McNemar's Tests (per-GT-box correct/incorrect) ===")
    for a, b in pairs:
        key = f"{a} vs {b}"
        result = mcnemar_test(per_gt_outcomes[a], per_gt_outcomes[b])
        results["mcnemar"][key] = result
        print(f"  {key}: {result['contingency_table']}, p={result['p_value']}")

    out_path = STATS_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()