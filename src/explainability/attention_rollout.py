from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import RTDETR

from gradcam import EigenCAM, _crop_to_bbox, _letterbox, _unletterbox_cam


print("=" * 80, flush=True)
print("RT-DETR EXPLAINABILITY", flush=True)
print("=" * 80, flush=True)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
IMAGE_DIR = ROOT / "data" / "processed" / "test" / "images"
LABEL_DIR = ROOT / "data" / "processed" / "test" / "labels"
OUTPUT_DIR = RESULTS_DIR / "explainability" / "rtdetr"

IMGSZ = 640
CONF_THRESHOLD = 0.25
MAX_SAMPLES = 3
MIN_IOU = 0.50
PADDING_RATIO = 0.60

CLASS_NAMES = [
    "cracking",
    "Layer_shifting",
    "Off_platform",
    "Stringing",
    "Wrapping",
]


def find_model():
    print("\nSearching for RT-DETR checkpoint...", flush=True)

    candidates = []

    for path in RESULTS_DIR.rglob("*.pt"):
        name = path.name.lower()
        full = str(path).lower()

        if "rtdetr" in full:
            candidates.append(path)

    if not candidates:
        print("No RT-DETR .pt file found inside results/", flush=True)

        print("\nAll .pt files found:", flush=True)

        all_models = list(RESULTS_DIR.rglob("*.pt"))

        if not all_models:
            print("NO .pt FILE FOUND.", flush=True)
        else:
            for path in all_models:
                print(f"  {path}", flush=True)

        raise FileNotFoundError(
            "Could not find RT-DETR checkpoint."
        )

    print("\nRT-DETR checkpoints found:", flush=True)

    for path in candidates:
        print(f"  {path}", flush=True)

    best = None

    for path in candidates:
        if path.name.lower() == "best.pt":
            best = path
            break

    if best is None:
        best = candidates[0]

    print(f"\nUsing model:\n{best}", flush=True)

    return best


def load_gt(label_path, width, height):
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            annotations.append({
                "class": cls,
                "bbox": np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32,
                ),
            })

    return annotations


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def find_good_samples(model):
    print("\n" + "=" * 80, flush=True)
    print("SEARCHING TEST IMAGES", flush=True)
    print("=" * 80, flush=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        image_paths = sorted(IMAGE_DIR.glob("*.png"))

    print(
        f"Test images found: {len(image_paths)}",
        flush=True,
    )

    candidates = []

    for index, image_path in enumerate(image_paths):
        print(
            f"\rChecking image {index + 1}/{len(image_paths)}...",
            end="",
            flush=True,
        )

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            LABEL_DIR / f"{image_path.stem}.txt"
        )

        gt = load_gt(
            label_path,
            width,
            height,
        )

        if not gt:
            continue

        try:
            results = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                device="cpu",
                verbose=False,
            )
        except Exception:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        if len(result.boxes) == 0:
            continue

        pred_boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        pred_scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        pred_classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        best_match = None

        for pred_box, score, pred_class in zip(
            pred_boxes,
            pred_scores,
            pred_classes,
        ):
            for gt_item in gt:
                if pred_class != gt_item["class"]:
                    continue

                current_iou = iou(
                    pred_box,
                    gt_item["bbox"],
                )

                if current_iou < MIN_IOU:
                    continue

                candidate = {
                    "image": image_path,
                    "bbox": pred_box,
                    "score": float(score),
                    "class": int(pred_class),
                    "iou": float(current_iou),
                }

                if (
                    best_match is None
                    or current_iou > best_match["iou"]
                ):
                    best_match = candidate

        if best_match is not None:
            candidates.append(best_match)

    print("\n", flush=True)

    candidates.sort(
        key=lambda x: (
            x["iou"],
            x["score"],
        ),
        reverse=True,
    )

    print(
        f"Good detections found: {len(candidates)}",
        flush=True,
    )

    if not candidates:
        raise RuntimeError(
            "No suitable RT-DETR detections found."
        )

    selected = []
    used_images = set()
    used_classes = set()

    for candidate in candidates:
        if candidate["class"] in used_classes:
            continue

        selected.append(candidate)
        used_images.add(candidate["image"])
        used_classes.add(candidate["class"])

        if len(selected) == MAX_SAMPLES:
            break

    if len(selected) < MAX_SAMPLES:
        for candidate in candidates:
            if candidate["image"] in used_images:
                continue

            selected.append(candidate)
            used_images.add(candidate["image"])

            if len(selected) == MAX_SAMPLES:
                break

    print("\n" + "=" * 80, flush=True)
    print("SELECTED SAMPLES", flush=True)
    print("=" * 80, flush=True)

    for i, item in enumerate(selected, 1):
        class_name = (
            CLASS_NAMES[item["class"]]
            if item["class"] < len(CLASS_NAMES)
            else str(item["class"])
        )

        print(f"\nSample {i}", flush=True)
        print(
            f"Image      : {item['image']}",
            flush=True,
        )
        print(
            f"Class      : {class_name}",
            flush=True,
        )
        print(
            f"Confidence : {item['score']:.4f}",
            flush=True,
        )
        print(
            f"IoU        : {item['iou']:.4f}",
            flush=True,
        )

    return selected


def generate_cam(model, item, sample_number):
    image_path = item["image"]
    bbox = item["bbox"]

    print(
        f"\nGenerating CAM for sample {sample_number}...",
        flush=True,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read:\n{image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    display_img = _crop_to_bbox(
        image_rgb,
        bbox,
        PADDING_RATIO,
    )

    letterboxed, scale, pad_x, pad_y = _letterbox(
        display_img,
        IMGSZ,
    )

    input_tensor = (
        torch.from_numpy(letterboxed)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )

    device = next(
        model.model.parameters()
    ).device

    input_tensor = input_tensor.to(device)

    target_layer = model.model.model[-5]

    print(
        f"Target layer: model.model.model[-5]",
        flush=True,
    )

    cam_extractor = EigenCAM(
        model.model,
        target_layer,
    )

    grayscale_cam = cam_extractor(
        input_tensor
    )

    cam_extractor.remove_hook()

    grayscale_cam_full = cv2.resize(
        grayscale_cam,
        (IMGSZ, IMGSZ),
    )

    grayscale_cam_resized = _unletterbox_cam(
        grayscale_cam_full,
        scale,
        pad_x,
        pad_y,
        display_img.shape[0],
        display_img.shape[1],
    )

    heatmap = cv2.applyColorMap(
        np.uint8(
            255 * grayscale_cam_resized
        ),
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        display_img,
        0.6,
        heatmap_rgb,
        0.4,
        0,
    )

    bx1, by1, bx2, by2 = bbox

    x1 = int(bx1 - max(0, bx1 - (bbox[2] - bbox[0]) * PADDING_RATIO))
    y1 = int(by1 - max(0, by1 - (bbox[3] - bbox[1]) * PADDING_RATIO))
    x2 = int(bx2 - max(0, bx1 - (bbox[2] - bbox[0]) * PADDING_RATIO))
    y2 = int(by2 - max(0, by1 - (bbox[3] - bbox[1]) * PADDING_RATIO))

    x1 = max(0, min(x1, overlay.shape[1] - 1))
    y1 = max(0, min(y1, overlay.shape[0] - 1))
    x2 = max(0, min(x2, overlay.shape[1] - 1))
    y2 = max(0, min(y2, overlay.shape[0] - 1))

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    class_id = item["class"]

    class_name = (
        CLASS_NAMES[class_id]
        if class_id < len(CLASS_NAMES)
        else str(class_id)
    )

    label = (
        f"{class_name} "
        f"{item['score']:.2f}"
    )

    cv2.putText(
        overlay,
        label,
        (
            max(5, x1),
            max(20, y1 - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"sample_{sample_number}_"
            f"{image_path.stem}.jpg"
        )
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print(
        f"SAVED: {output_path}",
        flush=True,
    )


def main():
    print(
        f"\nProject root:\n{ROOT}",
        flush=True,
    )

    print(
        f"\nImage directory:\n{IMAGE_DIR}",
        flush=True,
    )

    print(
        f"\nLabel directory:\n{LABEL_DIR}",
        flush=True,
    )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory does not exist:\n{LABEL_DIR}"
        )

    model_path = find_model()

    print(
        "\nLoading RT-DETR model...",
        flush=True,
    )

    model = RTDETR(
        str(model_path)
    )

    print(
        "RT-DETR loaded successfully.",
        flush=True,
    )

    print(
        f"Model type: {type(model)}",
        flush=True,
    )

    model.model.eval()

    samples = find_good_samples(model)

    print(
        "\nStarting Explainability...",
        flush=True,
    )

    for number, item in enumerate(
        samples,
        start=1,
    ):
        try:
            generate_cam(
                model,
                item,
                number,
            )
        except Exception as exc:
            print(
                f"\nFAILED SAMPLE {number}",
                flush=True,
            )
            print(
                f"Image: {item['image']}",
                flush=True,
            )
            print(
                f"Error: {type(exc).__name__}: {exc}",
                flush=True,
            )

    print("\n" + "=" * 80, flush=True)
    print("RT-DETR EXPLAINABILITY FINISHED", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()