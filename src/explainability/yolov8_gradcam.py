from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_NAMES = ["cracking", "Layer_shifting", "Off_platform", "Stringing", "Wrapping"]


class OutputTensorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class DetectionScoreTarget:
    def __init__(self, class_idx, anchor_idx):
        self.class_idx = int(class_idx)
        self.anchor_idx = int(anchor_idx)

    def __call__(self, model_output):
        if model_output.ndim == 3:
            model_output = model_output[0]
        return model_output[4 + self.class_idx, self.anchor_idx]


class YOLOv8GradCAM:
    def __init__(self, model, device="cpu", conf_threshold=0.25, imgsz=640):
        self.model = model
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.model.to(self.device)
        self.model.model.eval()

    def _find_test_images(self):
        root = Path(__file__).resolve().parents[2]
        image_dir = root / "data" / "processed" / "test" / "images"
        label_dir = root / "data" / "processed" / "test" / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found:\n{image_dir}")

        images = sorted(
            list(image_dir.glob("*.jpg")) +
            list(image_dir.glob("*.jpeg")) +
            list(image_dir.glob("*.png"))
        )

        valid = []
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists() and label_path.read_text().strip():
                valid.append((image_path, label_path))

        if not valid:
            raise FileNotFoundError(f"No image with annotation found in:\n{image_dir}")

        return valid

    def _read_annotation(self, label_path):
        annotations = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                annotations.append({
                    "class": int(float(parts[0])),
                    "xc": float(parts[1]),
                    "yc": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4])
                })

        return annotations

    def _xywh_to_xyxy(self, ann, width, height):
        xc = ann["xc"] * width
        yc = ann["yc"] * height
        w = ann["w"] * width
        h = ann["h"] * height

        return np.array([
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2
        ], dtype=np.float32)

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _select_image_annotation(self, image_path):
        label_path = (
            image_path.parent.parent /
            "labels" /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(f"Annotation not found:\n{label_path}")

        annotations = self._read_annotation(label_path)

        if not annotations:
            raise ValueError(f"No annotation found:\n{label_path}")

        return label_path, annotations

    def _get_scale_layer(self, torch_model, anchor_idx):
        if anchor_idx < 6400:
            scale_idx = 0
        elif anchor_idx < 8000:
            scale_idx = 1
        else:
            scale_idx = 2

        return torch_model.model[-1].cv3[scale_idx][-2]

    def _letterbox_pad(self, original_shape):
        h, w = original_shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)

        new_w = round(w * r)
        new_h = round(h * r)

        dw = self.imgsz - new_w
        dh = self.imgsz - new_h

        dw /= 2
        dh /= 2

        top = round(dh - 0.1)
        bottom = round(dh + 0.1)
        left = round(dw - 0.1)
        right = round(dw + 0.1)

        return top, bottom, left, right

    def _find_raw_target(self, raw):
        class_scores = raw[4:, :]
        best_score, best_class = class_scores.max(dim=0)

        anchor_idx = int(best_score.argmax().item())
        class_idx = int(best_class[anchor_idx].item())
        score = float(best_score[anchor_idx].item())

        return anchor_idx, class_idx, score

    def generate(self, image_path=None, detection_index=0, output_path=None):
        if image_path is None:
            image_path = self._find_test_images()[0][0]

        image_path = Path(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image:\n{image_path}"
            )

        label_path, annotations = self._select_image_annotation(image_path)

        h, w = image.shape[:2]

        print(f"Selected image: {image_path}")
        print(f"Annotation: {label_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        letterbox = LetterBox(
            (self.imgsz, self.imgsz),
            auto=False,
            stride=32
        )

        letterboxed = letterbox(image=rgb)

        tensor = (
            torch.from_numpy(letterboxed)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(self.device)

        rgb_float = letterboxed.astype(np.float32) / 255.0

        torch_model = self.model.model
        torch_model.eval()

        wrapped = OutputTensorWrapper(torch_model)

        with torch.no_grad():
            raw_output = wrapped(tensor)

        raw = raw_output[0] if raw_output.ndim == 3 else raw_output

        anchor_idx, class_idx, confidence = self._find_raw_target(raw)

        target_layer = self._get_scale_layer(
            torch_model,
            anchor_idx
        )

        print(f"Raw anchor: {anchor_idx}")
        print(f"Class: {class_idx}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Target layer: {target_layer}")

        torch_model.requires_grad_(True)

        target = DetectionScoreTarget(
            class_idx,
            anchor_idx
        )

        cam = GradCAM(
            model=wrapped,
            target_layers=[target_layer]
        )

        with torch.enable_grad():
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[target]
            )[0]

        top, bottom, left, right = self._letterbox_pad(
            image.shape
        )

        grayscale_cam = grayscale_cam.copy()

        if top:
            grayscale_cam[:top, :] = 0

        if bottom:
            grayscale_cam[-bottom:, :] = 0

        if left:
            grayscale_cam[:, :left] = 0

        if right:
            grayscale_cam[:, -right:] = 0

        visualization = show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.65
        )

        cx, cy, bw, bh = (
            raw[:4, anchor_idx]
            .detach()
            .cpu()
            .tolist()
        )

        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        class_name = (
            CLASS_NAMES[class_idx]
            if class_idx < len(CLASS_NAMES)
            else str(class_idx)
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            visualization,
            label,
            (max(4, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if output_path is None:
            root = Path(__file__).resolve().parents[2]

            output_path = (
                root /
                "results" /
                "explainability" /
                f"gradcam_{image_path.stem}.jpg"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR
            )
        )

        print(f"Saved: {output_path}")

        return {
            "prediction_class": class_idx,
            "prediction_class_name": class_name,
            "prediction_confidence": confidence,
            "raw_prediction_index": anchor_idx,
            "target_layer": str(target_layer),
            "output_path": str(output_path)
        }