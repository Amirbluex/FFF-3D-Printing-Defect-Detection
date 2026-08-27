from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import RTDETR
from ultralytics.data.augment import LetterBox


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "processed" / "test" / "images"
LABEL_DIR = ROOT / "data" / "processed" / "test" / "labels"
MODEL_PATH = ROOT / "results" / "rtdetr" / "run" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "results" / "explainability"


def find_image():
    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text().strip():
            return image_path
    raise FileNotFoundError(
        f"No image with annotation found in:\n{IMAGE_DIR}"
    )


def attention_to_heatmap(sampling_locations, attention_weights, spatial_shapes, points_per_level, top_ratio=0.20):
    locations = np.asarray(sampling_locations, dtype=np.float32)
    weights = np.asarray(attention_weights, dtype=np.float32)

    combined = np.zeros((640, 640), dtype=np.float32)
    start = 0

    for (height, width), n_points in zip(spatial_shapes, points_per_level):
        height = int(height)
        width = int(width)

        level_weights = weights[:, start:start + n_points]
        threshold = np.quantile(level_weights, 1.0 - top_ratio)
        level_weights = np.where(level_weights >= threshold, level_weights, 0.0)

        level = np.zeros((height, width), dtype=np.float32)

        for head in range(locations.shape[0]):
            for point in range(start, start + n_points):
                weight = float(level_weights[head, point - start])

                if weight <= 0:
                    continue

                x = float(locations[head, point, 0]) * width - 0.5
                y = float(locations[head, point, 1]) * height - 0.5

                if x < -1 or x > width or y < -1 or y > height:
                    continue

                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                dx = x - x0
                dy = y - y0

                for xx, yy, factor in (
                    (x0, y0, (1 - dx) * (1 - dy)),
                    (x0 + 1, y0, dx * (1 - dy)),
                    (x0, y0 + 1, (1 - dx) * dy),
                    (x0 + 1, y0 + 1, dx * dy),
                ):
                    if 0 <= xx < width and 0 <= yy < height:
                        level[yy, xx] += weight * factor

        if level.max() > 0:
            level /= level.max()

        combined += cv2.resize(
            level,
            (640, 640),
            interpolation=cv2.INTER_LINEAR,
        )

        start += n_points

    combined = cv2.GaussianBlur(
        combined,
        (0, 0),
        sigmaX=5,
        sigmaY=5,
    )

    combined -= combined.min()
    combined /= combined.max() + 1e-8

    return combined.astype(np.float32)


def letterbox_padding(
    original_height,
    original_width,
    size=640,
):
    ratio = min(
        size / original_height,
        size / original_width,
    )

    new_width = round(original_width * ratio)
    new_height = round(original_height * ratio)

    dw = size - new_width
    dh = size - new_height

    dw /= 2
    dh /= 2

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)

    return top, bottom, left, right


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RT-DETR checkpoint not found:\n{MODEL_PATH}"
        )

    image_path = (
    IMAGE_DIR / "Image_20240103134846635.jpg"
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "rtdetr_attention_test.jpg"
    )

    print("=" * 80)
    print("RT-DETR Detection-Specific Attention")
    print("=" * 80)
    print("Model:", MODEL_PATH)
    print("Image:", image_path)
    print("Output:", output_path)
    print()

    model = RTDETR(str(MODEL_PATH))
    torch_model = model.model
    torch_model.eval()

    letterbox = LetterBox(
        (640, 640),
        auto=False,
        stride=32,
    )

    captured = {}

    cross_attn = (
        torch_model
        .model[-1]
        .decoder
        .layers[-1]
        .cross_attn
    )

    def hook(module, inputs, output):
        captured["inputs"] = inputs

    handle = cross_attn.register_forward_hook(
        hook
    )

    try:
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image:\n{image_path}"
            )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = (
            rgb.shape[:2]
        )

        letterboxed = letterbox(
            image=rgb
        )

        tensor = (
            torch.from_numpy(letterboxed)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        )

        rgb_float = (
            letterboxed.astype(np.float32)
            / 255.0
        )

        with torch.no_grad():
            outputs = torch_model(
                tensor
            )

        detections = outputs[0][0]

        print(
            "Raw detection shape:",
            tuple(detections.shape),
        )

        query_idx = 0

        (
            cx,
            cy,
            bw,
            bh,
            score,
            class_id,
        ) = detections[
            query_idx
        ].detach().cpu().tolist()

        print(
            "Selected query:",
            query_idx,
        )
        print(
            "Confidence:",
            f"{score:.4f}",
        )
        print(
            "Class ID:",
            int(class_id),
        )

        if hasattr(model, "names"):
            class_name = model.names[
                int(class_id)
            ]
        else:
            class_name = str(
                int(class_id)
            )

        print(
            "Class:",
            class_name,
        )

        query, reference, _, value_shapes, _ = (
            captured["inputs"]
        )

        batch_size, num_queries = (
            query.shape[:2]
        )

        num_levels = (
            cross_attn.n_levels
        )
        num_points = (
            cross_attn.n_points
        )
        total_points = (
            num_levels * num_points
        )

        offsets = (
            cross_attn
            .sampling_offsets(query)
            .view(
                batch_size,
                num_queries,
                cross_attn.n_heads,
                total_points,
                2,
            )
        )

        weights = torch.softmax(
            cross_attn
            .attention_weights(query)
            .view(
                batch_size,
                num_queries,
                cross_attn.n_heads,
                total_points,
            ),
            dim=-1,
        )

        if reference.shape[-1] == 4:
            locations = (
                reference[
                    :, :, None, :, :2
                ]
                + offsets
                / num_points
                * reference[
                    :, :, None, :, 2:
                ]
                * 0.5
            )
        else:
            normalizer = torch.as_tensor(
                value_shapes,
                dtype=query.dtype,
            ).flip(-1)

            normalizer = (
                normalizer[
                    :, None, :
                ]
                .expand(
                    -1,
                    num_points,
                    -1,
                )
                .reshape(
                    total_points,
                    2,
                )
            )

            locations = (
                reference[
                    :, :, None, :, :
                ]
                + offsets
                / normalizer
            )

        spatial_shapes = [
            (
                int(height),
                int(width),
            )
            for height, width
            in value_shapes
        ]

        heatmap = attention_to_heatmap(
            locations[
                0,
                query_idx,
            ]
            .detach()
            .cpu()
            .numpy(),
            weights[
                0,
                query_idx,
            ]
            .detach()
            .cpu()
            .numpy(),
            spatial_shapes,
            [num_points] * num_levels,
        )

        top, bottom, left, right = (
            letterbox_padding(
                original_height,
                original_width,
                640,
            )
        )

        if top:
            heatmap[:top, :] = 0
        if bottom:
            heatmap[-bottom:, :] = 0
        if left:
            heatmap[:, :left] = 0
        if right:
            heatmap[:, -right:] = 0

        heatmap_uint8 = np.uint8(
            np.clip(
                heatmap,
                0,
                1,
            )
            * 255
        )

        colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB,
        )

        visualization = cv2.addWeighted(
            letterboxed,
            0.65,
            colored,
            0.35,
            0,
        )

        if max(
            abs(cx),
            abs(cy),
            abs(bw),
            abs(bh),
        ) <= 2.0:
            cx *= 640
            cy *= 640
            bw *= 640
            bh *= 640

        x1 = int(
            cx - bw / 2
        )
        y1 = int(
            cy - bh / 2
        )
        x2 = int(
            cx + bw / 2
        )
        y2 = int(
            cy + bh / 2
        )

        x1 = max(
            0,
            min(639, x1),
        )
        y1 = max(
            0,
            min(639, y1),
        )
        x2 = max(
            0,
            min(639, x2),
        )
        y2 = max(
            0,
            min(639, y2),
        )

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        label = (
            f"{class_name} "
            f"{score:.2f}"
        )

        cv2.putText(
            visualization,
            label,
            (
                max(4, x1),
                max(18, y1 - 7),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR,
            ),
        )

        print()
        print(
            "Bounding box:",
            (x1, y1, x2, y2),
        )
        print(
            "Target layer:",
            "decoder.layers[-1].cross_attn",
        )
        print(
            "Feature levels:",
            spatial_shapes,
        )
        print(
            "Attention heads:",
            cross_attn.n_heads,
        )
        print()
        print(
            "Saved:",
            output_path,
        )
        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)

    finally:
        handle.remove()


if __name__ == "__main__":
    main()