"""
Applies the base paper's preprocessing/augmentation pipeline:
1. Grayscale conversion (replicated to 3 channels) -- applied to ALL
   splits (train/val/test), since this is preprocessing, not augmentation.
2. Elastic transformation -- applied to TRAIN ONLY.
3. Rotation +/-45 degrees -- applied to TRAIN ONLY.

Deviation from the paper, intentionally: the paper split its 1176
images (already augmented) into 8:1:1, meaning augmented variants of
the same source photo could land in different splits (train vs test).
We avoid this leakage by augmenting ONLY the train split, after the
split already happened. This is documented as a methodological
improvement in the report, not a silent deviation.

Elastic transform note: bounding boxes are left unchanged under
elastic transform, since it's a smooth, spatially-local per-pixel
deformation (standard simplification -- an exact box warp would
require tracking a deformed mask, out of scope here). Rotation DOES
correctly transform bounding boxes (recomputed as the tightest
axis-aligned box around the rotated corners).
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, map_coordinates

PROCESSED_DIR = Path("data/processed")
SPLITS_TO_GRAYSCALE_ONLY = ["val", "test"]
SPLIT_TO_AUGMENT = "train"

ELASTIC_ALPHA = 1
ELASTIC_SIGMA = 50
ROTATION_ANGLES = [-45, 45]

SEED = 42
rng = np.random.RandomState(SEED)


def to_grayscale_3ch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def elastic_transform(img):
    """
    Classic Simard et al. elastic transform: smooth random displacement
    field applied per pixel. Bounding boxes are NOT adjusted.
    """
    shape = img.shape[:2]
    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), ELASTIC_SIGMA) * ELASTIC_ALPHA
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), ELASTIC_SIGMA) * ELASTIC_ALPHA

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    if img.ndim == 3:
        channels = [cv2.remap(img[:, :, c], map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
                    for c in range(img.shape[2])]
        return np.stack(channels, axis=2)
    else:
        return cv2.remap(img, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def rotate_image_and_boxes(img, boxes_yolo, angle_deg):
    """
    Rotates the image around its center by angle_deg, keeping the same
    canvas size. Recomputes each YOLO box as the tightest axis-aligned
    box around its 4 corners after rotation.

    boxes_yolo: list of (class_id, x_center, y_center, w, h), all
    normalized [0,1].

    Returns: (rotated_img, new_boxes_yolo)
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated_img = cv2.warpAffine(img, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT)

    new_boxes = []
    for class_id, xc, yc, bw, bh in boxes_yolo:
        # normalized -> absolute pixel corners
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ])

        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated_corners = (M @ corners_h.T).T  # (4, 2)

        new_x1 = np.clip(rotated_corners[:, 0].min(), 0, w)
        new_y1 = np.clip(rotated_corners[:, 1].min(), 0, h)
        new_x2 = np.clip(rotated_corners[:, 0].max(), 0, w)
        new_y2 = np.clip(rotated_corners[:, 1].max(), 0, h)

        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        if new_w <= 1 or new_h <= 1:
            continue  # box rotated entirely out of frame, drop it

        new_xc = (new_x1 + new_x2) / 2 / w
        new_yc = (new_y1 + new_y2) / 2 / h
        new_bw = new_w / w
        new_bh = new_h / h

        new_boxes.append((class_id, new_xc, new_yc, new_bw, new_bh))

    return rotated_img, new_boxes


def read_yolo_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            boxes.append((class_id, xc, yc, bw, bh))
    return boxes


def write_yolo_labels(label_path, boxes):
    lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in boxes]
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def process_grayscale_only_split(split_name):
    images_dir = PROCESSED_DIR / split_name / "images"
    count = 0
    for img_path in images_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray_img = to_grayscale_3ch(img)
        cv2.imwrite(str(img_path), gray_img)  # overwrite in place
        count += 1
    print(f"{split_name}: grayscale-converted {count} images (in place)")


def process_train_split():
    images_dir = PROCESSED_DIR / SPLIT_TO_AUGMENT / "images"
    labels_dir = PROCESSED_DIR / SPLIT_TO_AUGMENT / "labels"

    image_paths = sorted(images_dir.glob("*.*"))
    base_count = len(image_paths)
    generated = 0

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = read_yolo_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Step 1: grayscale (base preprocessing), overwrite original in place
        gray_img = to_grayscale_3ch(img)
        cv2.imwrite(str(img_path), gray_img)

        # Step 2: elastic transform variant (boxes unchanged)
        elastic_img = elastic_transform(gray_img)
        elastic_img_path = images_dir / f"{img_path.stem}_elastic{img_path.suffix}"
        elastic_label_path = labels_dir / f"{img_path.stem}_elastic.txt"
        cv2.imwrite(str(elastic_img_path), elastic_img)
        write_yolo_labels(elastic_label_path, boxes)
        generated += 1

        # Step 3 & 4: rotation variants (boxes recomputed)
        for angle in ROTATION_ANGLES:
            rot_img, rot_boxes = rotate_image_and_boxes(gray_img, boxes, angle)
            sign = "ccw45" if angle < 0 else "cw45"
            rot_img_path = images_dir / f"{img_path.stem}_rot{sign}{img_path.suffix}"
            rot_label_path = labels_dir / f"{img_path.stem}_rot{sign}.txt"
            cv2.imwrite(str(rot_img_path), rot_img)
            write_yolo_labels(rot_label_path, rot_boxes)
            generated += 1

    print(f"train: {base_count} originals (grayscale-converted in place) + "
          f"{generated} new augmented images generated "
          f"(total train images now: {base_count + generated})")


def main():
    for split_name in SPLITS_TO_GRAYSCALE_ONLY:
        process_grayscale_only_split(split_name)

    process_train_split()

    print("\nDone. Re-run verify_dataset.py to sanity-check the updated train split.")


if __name__ == "__main__":
    main()