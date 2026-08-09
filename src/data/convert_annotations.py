"""
Converts the COCO-format annotations (output of merge_and_split.py) into
YOLO-format label files, and generates the data.yaml file Ultralytics
needs for training.

YOLO format: one .txt file per image, same basename, containing one line
per bounding box: <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
all normalized to [0, 1] relative to image width/height.
"""

import json
from pathlib import Path
import yaml

PROCESSED_DIR = Path("data/processed")
CONFIG_PATH = Path("configs/dataset_config.yaml")
SPLITS = ["train", "val", "test"]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def convert_split(split_name):
    split_dir = PROCESSED_DIR / split_name
    json_path = split_dir / "_annotations.coco.json"

    if not json_path.exists():
        print(f"ERROR: {json_path} not found -- skipping {split_name}")
        return None

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # COCO category ids are NOT guaranteed to start at 0 or be contiguous
    # in a way YOLO expects -- build an explicit 0-indexed mapping, sorted
    # by category id for a stable, reproducible ordering.
    categories_sorted = sorted(coco_data["categories"], key=lambda c: c["id"])
    coco_id_to_yolo_id = {c["id"]: i for i, c in enumerate(categories_sorted)}
    yolo_class_names = [c["name"] for c in categories_sorted]

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    written, empty = 0, 0

    for img in coco_data["images"]:
        img_w, img_h = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        label_path = labels_dir / f"{stem}.txt"

        anns = anns_by_image.get(img["id"], [])
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: top-left x,y + width,height

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            yolo_class_id = coco_id_to_yolo_id[ann["category_id"]]
            lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} "
                         f"{w_norm:.6f} {h_norm:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if lines:
            written += 1
        else:
            empty += 1

        # Move the actual image into an images/ subfolder, which is the
        # layout Ultralytics expects (images/ and labels/ as siblings
        # with matching filenames).
        src_img_path = split_dir / img["file_name"]
        dst_img_path = images_dir / img["file_name"]
        if src_img_path.exists() and not dst_img_path.exists():
            src_img_path.rename(dst_img_path)

    print(f"{split_name}: {written} label files written, {empty} images "
          f"with no annotations (should be 0 -- investigate if not)")

    return yolo_class_names


def write_data_yaml(class_names):
    data_yaml = {
        "train": str((PROCESSED_DIR / "train" / "images").resolve()),
        "val": str((PROCESSED_DIR / "val" / "images").resolve()),
        "test": str((PROCESSED_DIR / "test" / "images").resolve()),
        "nc": len(class_names),
        "names": class_names,
    }

    out_path = PROCESSED_DIR / "data.yaml"
    with open(out_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {out_path}")
    print(f"Classes ({len(class_names)}): {class_names}")


def main():
    class_names = None
    for split_name in SPLITS:
        result = convert_split(split_name)
        if result is not None:
            class_names = result  # same across all splits, just keep the last one

    if class_names:
        write_data_yaml(class_names)
    else:
        print("ERROR: no splits converted successfully -- data.yaml not written")


if __name__ == "__main__":
    main()