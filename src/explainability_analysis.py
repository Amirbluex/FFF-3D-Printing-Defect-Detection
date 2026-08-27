import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import subprocess
from ultralytics import YOLO
from src.explainability.yolov8_gradcam import YOLOv8GradCAM
from src.explainability.deimv2_attention import DEIMv2Attention

import subprocess
from ultralytics import YOLO
from src.explainability.yolov8_gradcam import YOLOv8GradCAM
from src.explainability.deimv2_attention import DEIMv2Attention

IMAGE_DIR = ROOT / "data" / "processed" / "test" / "images"
LABEL_DIR = ROOT / "data" / "processed" / "test" / "labels"
OUTPUT_DIR = ROOT / "results" / "explainability"

YOLO_WEIGHTS = ROOT / "results" / "baseline_yolov8" / "run" / "weights" / "best.pt"
RTDETR_SCRIPT = ROOT / "src" / "explainability" / "attention_rollout.py"

DEIM_CANDIDATES = [
    ROOT / "results" / "deimv2" / "checkpoint-2688",
    ROOT / "results" / "dino" / "run" / "checkpoint-2688",
    ROOT / "results" / "DEIMv2" / "checkpoint-2688",
]

YOLO_OUTPUT = OUTPUT_DIR / "yolov8"
RTDETR_OUTPUT = OUTPUT_DIR / "rtdetr"
DEIM_OUTPUT = OUTPUT_DIR / "deimv2"

for directory in [YOLO_OUTPUT, RTDETR_OUTPUT, DEIM_OUTPUT]:
    directory.mkdir(parents=True, exist_ok=True)


def get_test_images(n=3):
    selected = []

    for image_path in sorted(IMAGE_DIR.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            selected.append((image_path, label_path))

        if len(selected) == n:
            break

    if not selected:
        raise FileNotFoundError(
            f"No annotated images found in: {IMAGE_DIR}"
        )

    return selected


def find_deim_checkpoint():
    for checkpoint in DEIM_CANDIDATES:
        if checkpoint.exists():
            return checkpoint

    candidates = []

    for path in (ROOT / "results").rglob("*"):
        if not path.is_dir():
            continue

        if path.name.startswith("checkpoint-") and (
            (path / "config.json").exists()
            or (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
        ):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "DEIMv2 checkpoint could not be found inside results/"
        )

    return sorted(candidates)[-1]


def run_yolov8(images):
    print("\n" + "=" * 80)
    print("YOLOv8 EXPLAINABILITY")
    print("=" * 80)

    model = YOLO(str(YOLO_WEIGHTS))

    gradcam = YOLOv8GradCAM(
        model=model,
        device="cpu",
        conf_threshold=0.25,
        imgsz=640,
    )

    for index, (image_path, _) in enumerate(images, 1):
        print(f"\nYOLOv8 {index}/{len(images)}")
        print(f"Image: {image_path}")

        output_path = YOLO_OUTPUT / f"{image_path.stem}.jpg"

        try:
            result = gradcam.generate(
                image_path=str(image_path),
                detection_index=0,
                output_path=str(output_path),
            )

            print(f"Class: {result.get('prediction_class_name')}")
            print(f"Confidence: {result.get('prediction_confidence')}")
            print(f"IoU: {result.get('raw_prediction_match_iou')}")
            print(f"Output: {output_path}")

        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")


def run_rtdetr():
    print("\n" + "=" * 80)
    print("RT-DETR EXPLAINABILITY")
    print("=" * 80)

    if not RTDETR_SCRIPT.exists():
        print(f"FAILED: RT-DETR script not found:\n{RTDETR_SCRIPT}")
        return

    try:
        result = subprocess.run(
            [sys.executable, str(RTDETR_SCRIPT)],
            cwd=str(ROOT),
            capture_output=False,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"\nRT-DETR finished with exit code "
                f"{result.returncode}"
            )
        else:
            print("\nRT-DETR explainability completed.")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


def run_deimv2(images):
    print("\n" + "=" * 80)
    print("DEIMv2 EXPLAINABILITY")
    print("=" * 80)

    try:
        checkpoint = find_deim_checkpoint()

        print(f"Checkpoint: {checkpoint}")

        explainer = DEIMv2Attention(
            checkpoint=str(checkpoint),
            device="cpu",
        )

        for index, (image_path, _) in enumerate(images, 1):
            print(f"\nDEIMv2 {index}/{len(images)}")
            print(f"Image: {image_path}")

            output_path = DEIM_OUTPUT / f"{image_path.stem}.jpg"

            result = explainer.generate(
                image_path=str(image_path),
                output_path=str(output_path),
            )

            print(f"Query: {result['query_index']}")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Output: {output_path}")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


def main():
    print("=" * 80)
    print("FINAL EXPLAINABILITY PIPELINE")
    print("=" * 80)

    images = get_test_images(3)

    print("\nSelected annotated images:")
    for image_path, label_path in images:
        print(f"Image: {image_path}")
        print(f"Label: {label_path}")

    run_yolov8(images)
    run_rtdetr()
    run_deimv2(images)

    print("\n" + "=" * 80)
    print("ALL EXPLAINABILITY MODELS FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()