"""
Merges per-category COCO annotation exports from CVAT into a single
COCO dataset, then performs a stratified train/val/test split.

Two important details handled here, based on inspecting the actual
CVAT export structure:

1. Category IDs are already globally consistent across all 5 exported
   files (1=Cracking, 2=Layer_shifting, 3=Off_platform, 4=Stringing,
   5=Warping), since they all came from the same CVAT project. Only
   image_id / annotation_id need remapping to avoid collisions when
   merging files.

2. Some source images have augmented variants sitting alongside them
   (e.g. "Image_X.jpg", "Image_X_aug.jpg", "Image_X_original.jpg").
   These are NOT independent samples -- they're different versions of
   the same underlying photo. If left ungrouped, the random split
   could put e.g. "Image_X.jpg" in train and "Image_X_aug.jpg" in
   test, which is data leakage (the model would be tested on a near-
   duplicate of something it trained on, inflating metrics). This
   script groups all variants of the same base image together so they
   always land in the same split.
"""

import json
import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# CONFIG 
ANNOTATIONS_DIR = Path("data/annotations")   
RAW_IMAGES_DIR = Path("data/raw")             # original per-class image folders
OUTPUT_DIR = Path("data/processed")           # train/val/test destination
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

CATEGORY_JSON_MAP = {
    "Cracking": "cracking.json",
    "Layer_shifting": "layer_shifting.json",
    "Off_platform": "off_platform.json",
    "Stringing": "stringing.json",
    "Warping": "warping.json",
}

# Matches "_aug" or "_original" right before the file extension
VARIANT_SUFFIX_RE = re.compile(r"(_aug|_original)$")

random.seed(SEED)


def load_coco_json(path):
    with open(path, "r") as f:
        return json.load(f)


def base_image_key(file_name):
    """
    Strips _aug / _original suffixes (and extension) so that all
    variants of the same source photo map to the same grouping key.
    """
    stem = Path(file_name).stem
    stem = VARIANT_SUFFIX_RE.sub("", stem)
    return stem


def merge_annotations():
    """
    Merge the 5 per-category COCO files into one, remapping image_ids
    and annotation_ids to avoid collisions. Category IDs are trusted
    as-is since they're already consistent across files.
    """
    merged = {"images": [], "annotations": [], "categories": None}

    next_image_id = 1
    next_ann_id = 1

    # image_id (in merged set) -> base grouping key (for leakage-safe split)
    image_group_key = {}
    # image_id (in merged set) -> class name (for stratification)
    image_class_map = {}

    for class_name, json_filename in CATEGORY_JSON_MAP.items():
        json_path = ANNOTATIONS_DIR / json_filename
        if not json_path.exists():
            print(f"WARNING: {json_path} not found, skipping {class_name}")
            continue

        data = load_coco_json(json_path)

        if merged["categories"] is None:
            merged["categories"] = data["categories"]

        # Remap images
        old_imgid_to_new = {}
        for img in data["images"]:
            new_img = dict(img)
            old_id = img["id"]
            new_img["id"] = next_image_id
            old_imgid_to_new[old_id] = next_image_id

            merged["images"].append(new_img)
            image_group_key[next_image_id] = base_image_key(img["file_name"])
            image_class_map[next_image_id] = class_name

            next_image_id += 1

        # Remap annotations (category_id left as-is -- already consistent)
        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_imgid_to_new[ann["image_id"]]
            merged["annotations"].append(new_ann)
            next_ann_id += 1

        print(f"{class_name}: {len(data['images'])} images, "
              f"{len(data['annotations'])} boxes merged")

    return merged, image_group_key, image_class_map


def stratified_group_split(image_group_key, image_class_map):
    """
    Splits at the GROUP level (base image key), not the individual
    image level, so that all variants (plain / _aug / _original) of
    the same source photo always land in the same split. Stratified
    by class so each class keeps its proportional representation in
    train/val/test.
    """
    # group_key -> class (a group should only ever belong to one class,
    # since _aug/_original variants come from the same category export)
    group_to_class = {}
    group_to_image_ids = defaultdict(list)

    for img_id, group_key in image_group_key.items():
        group_to_image_ids[group_key].append(img_id)
        group_to_class[group_key] = image_class_map[img_id]

    # Sanity check: warn if a group somehow spans multiple classes
    for group_key, img_ids in group_to_image_ids.items():
        classes_in_group = {image_class_map[i] for i in img_ids}
        if len(classes_in_group) > 1:
            print(f"WARNING: group '{group_key}' spans multiple classes: "
                  f"{classes_in_group} -- check for a filename collision "
                  f"across category folders.")

    # Group keys by class for stratified splitting
    class_to_groups = defaultdict(list)
    for group_key, cls in group_to_class.items():
        class_to_groups[cls].append(group_key)

    split_assignment = {}
    for cls, group_keys in class_to_groups.items():
        group_keys = group_keys.copy()
        random.shuffle(group_keys)

        n = len(group_keys)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        n_images_train = n_images_val = n_images_test = 0

        for i, group_key in enumerate(group_keys):
            if i < n_train:
                split_name = "train"
            elif i < n_train + n_val:
                split_name = "val"
            else:
                split_name = "test"

            for img_id in group_to_image_ids[group_key]:
                split_assignment[img_id] = split_name

            if split_name == "train":
                n_images_train += len(group_to_image_ids[group_key])
            elif split_name == "val":
                n_images_val += len(group_to_image_ids[group_key])
            else:
                n_images_test += len(group_to_image_ids[group_key])

        print(f"{cls}: {n} source photos (groups) -> "
              f"train={n_train} groups/{n_images_train} images, "
              f"val={n_val} groups/{n_images_val} images, "
              f"test={n - n_train - n_val} groups/{n_images_test} images")

    return split_assignment


def write_split_files(merged, split_assignment):
    """
    Writes one COCO json per split, and copies image files into
    data/processed/{train,val,test}/
    """
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_img_ids = {i for i, s in split_assignment.items() if s == split}
        split_images = [img for img in merged["images"] if img["id"] in split_img_ids]
        split_anns = [ann for ann in merged["annotations"] if ann["image_id"] in split_img_ids]

        split_coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": merged["categories"],
        }

        out_json = split_dir / "_annotations.coco.json"
        with open(out_json, "w") as f:
            json.dump(split_coco, f, indent=2)

        copied, missing = 0, 0
        for img in split_images:
            filename = os.path.basename(img["file_name"])

            found = None
            for class_folder in RAW_IMAGES_DIR.iterdir():
                if not class_folder.is_dir():
                    continue
                candidate = class_folder / filename
                if candidate.exists():
                    found = candidate
                    break

            if found:
                shutil.copy(found, split_dir / filename)
                copied += 1
            else:
                missing += 1

        print(f"{split}: {len(split_images)} images, {len(split_anns)} boxes "
              f"-> copied {copied}, missing {missing}")
        if missing:
            print(f"  ({missing} images referenced in annotations were not "
                  f"found in data/raw/<class>/ -- check filenames)")


def main():
    print("Merging annotations...")
    merged, image_group_key, image_class_map = merge_annotations()

    n_groups = len(set(image_group_key.values()))
    print(f"\n{len(image_group_key)} total images belong to "
          f"{n_groups} unique source photo groups "
          f"(i.e. {len(image_group_key) - n_groups} are _aug/_original duplicates)")

    print("\nComputing group-aware stratified split...")
    split_assignment = stratified_group_split(image_group_key, image_class_map)

    print("\nWriting split files and copying images...")
    write_split_files(merged, split_assignment)

    print("\nDone. Check data/processed/{train,val,test}/")


if __name__ == "__main__":
    main()