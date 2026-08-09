"""
Verifies the processed dataset (output of merge_and_split.py) against
the expectations set in configs/dataset_config.yaml:
- correct number of classes
- image counts per split roughly match the 8:1:1 ratio
- every annotation references an image that actually exists on disk
- every bounding box is valid (positive width/height, within image bounds)
- class distribution per split is reported (useful for spotting imbalance)
- a random sample grid is saved so you can eyeball annotation quality
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml

CONFIG_PATH = Path("configs/dataset_config.yaml")
PROCESSED_DIR = Path("data/processed")
SPLITS = ["train", "val", "test"]
N_SAMPLES_TO_VISUALIZE = 20
OUTPUT_SAMPLE_GRID = Path("data/processed/verification_samples.png")

SEED = 42
random.seed(SEED)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_split(split_name):
    json_path = PROCESSED_DIR / split_name / "_annotations.coco.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def check_class_count(coco_data, expected_classes):
    cat_names = {c["name"] for c in coco_data["categories"]}
    expected_set = set(expected_classes)
    missing = expected_set - cat_names
    extra = cat_names - expected_set
    if missing:
        print(f"  WARNING: missing expected classes: {missing}")
    if extra:
        print(f"  WARNING: unexpected extra classes found: {extra}")
    if not missing and not extra:
        print(f"  OK: all {len(expected_classes)} expected classes present")


def check_images_exist(coco_data, split_dir):
    missing_files = []
    for img in coco_data["images"]:
        # Check images/ subfolder first (post-convert_annotations layout),
        # fall back to split root (pre-convert_annotations layout)
        img_path = split_dir / "images" / img["file_name"]
        if not img_path.exists():
            img_path = split_dir / img["file_name"]
        if not img_path.exists():
            missing_files.append(img["file_name"])
    if missing_files:
        print(f"  ERROR: {len(missing_files)} images referenced in annotations "
              f"are missing from disk:")
        for f in missing_files[:5]:
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
    else:
        print(f"  OK: all {len(coco_data['images'])} referenced images exist on disk")
    return missing_files


def check_bbox_validity(coco_data):
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}
    invalid = []
    for ann in coco_data["annotations"]:
        x, y, w, h = ann["bbox"]
        img_w, img_h = id_to_size.get(ann["image_id"], (None, None))

        if w <= 0 or h <= 0:
            invalid.append((ann["id"], "non-positive width/height"))
            continue
        if img_w is not None:
            if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                invalid.append((ann["id"], "bbox extends outside image bounds"))

    if invalid:
        print(f"  WARNING: {len(invalid)} invalid bounding boxes found:")
        for ann_id, reason in invalid[:5]:
            print(f"    - annotation id {ann_id}: {reason}")
        if len(invalid) > 5:
            print(f"    ... and {len(invalid) - 5} more")
    else:
        print(f"  OK: all {len(coco_data['annotations'])} bounding boxes are valid")
    return invalid


def report_class_distribution(coco_data, split_name):
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    counts = Counter(cat_id_to_name[ann["category_id"]] for ann in coco_data["annotations"])
    print(f"  Class distribution ({split_name}):")
    for cls_name in sorted(counts):
        print(f"    {cls_name}: {counts[cls_name]} boxes")
    return counts


def check_split_ratio(split_image_counts, expected_ratio):
    total = sum(split_image_counts.values())
    print("\nOverall split ratio check:")
    for split_name, count in split_image_counts.items():
        actual_pct = count / total * 100 if total else 0
        expected_pct = expected_ratio.get(split_name, 0) * 100
        print(f"  {split_name}: {count} images ({actual_pct:.1f}%) "
              f"-- expected ~{expected_pct:.1f}%")


def visualize_random_samples(all_split_data, n_samples):
    """
    Picks random images across all splits and draws their bounding boxes
    for a quick visual sanity check.
    """
    pool = []
    for split_name, (coco_data, split_dir) in all_split_data.items():
        cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
        anns_by_image = defaultdict(list)
        for ann in coco_data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for img in coco_data["images"]:
            if img["id"] in anns_by_image:
                pool.append((split_name, split_dir, img, anns_by_image[img["id"]], cat_id_to_name))

    if not pool:
        print("No annotated images found to visualize.")
        return

    sample = random.sample(pool, min(n_samples, len(pool)))
    n_cols = 5
    n_rows = (len(sample) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, (split_name, split_dir, img_info, anns, cat_id_to_name) in enumerate(sample):
        img_path = split_dir / "images" / img_info["file_name"]
        if not img_path.exists():
            img_path = split_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[i]
        ax.imshow(img)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                      edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            cls_name = cat_id_to_name[ann["category_id"]]
            ax.text(x, max(y - 5, 0), cls_name, color="lime", fontsize=8,
                    weight="bold", backgroundcolor="black")
        ax.set_title(f"{split_name}", fontsize=9)
        ax.axis("off")

    # hide unused subplots
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    OUTPUT_SAMPLE_GRID.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_SAMPLE_GRID, dpi=150)
    print(f"\nSaved visual verification grid to: {OUTPUT_SAMPLE_GRID}")


def main():
    config = load_config()
    expected_classes = config["dataset"]["classes"]
    expected_ratio = config["dataset"]["split_ratio"]

    all_split_data = {}
    split_image_counts = {}

    for split_name in SPLITS:
        print(f"\n=== Checking split: {split_name} ===")
        coco_data = load_split(split_name)
        if coco_data is None:
            continue

        split_dir = PROCESSED_DIR / split_name
        all_split_data[split_name] = (coco_data, split_dir)
        split_image_counts[split_name] = len(coco_data["images"])

        print(f"  Images: {len(coco_data['images'])}, Annotations: {len(coco_data['annotations'])}")
        check_class_count(coco_data, expected_classes)
        check_images_exist(coco_data, split_dir)
        check_bbox_validity(coco_data)
        report_class_distribution(coco_data, split_name)

    if split_image_counts:
        check_split_ratio(split_image_counts, expected_ratio)

    print("\nGenerating visual verification sample grid...")
    visualize_random_samples(all_split_data, N_SAMPLES_TO_VISUALIZE)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()