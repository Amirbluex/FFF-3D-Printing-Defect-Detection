from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Deimv2ForObjectDetection
from pytorch_grad_cam.utils.image import show_cam_on_image


class DEIMv2Attention:
    def __init__(self, checkpoint, processor_name="harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers", device="cpu"):
        self.device = torch.device(device)
        self.checkpoint = str(checkpoint)
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.model = Deimv2ForObjectDetection.from_pretrained(
            self.checkpoint,
            attn_implementation="eager"
        ).to(self.device)
        self.model.eval()
        self.cross_attn = self.model.model.decoder.layers[-1].encoder_attn

    @staticmethod
    def _heatmap(locations, weights, shapes, points_per_level, size=640):
        combined = np.zeros((size, size), dtype=np.float32)
        start = 0

        for (h, w), n_points in zip(shapes, points_per_level):
            h, w = int(h), int(w)
            level = np.zeros((h, w), dtype=np.float32)

            for head in range(locations.shape[0]):
                for point in range(start, start + n_points):
                    x = float(locations[head, point, 0]) * w - 0.5
                    y = float(locations[head, point, 1]) * h - 0.5

                    if x < -1 or x > w or y < -1 or y > h:
                        continue

                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    dx, dy = x - x0, y - y0
                    value = float(weights[head, point]) / locations.shape[0]

                    for xx, yy, factor in (
                        (x0, y0, (1 - dx) * (1 - dy)),
                        (x0 + 1, y0, dx * (1 - dy)),
                        (x0, y0 + 1, (1 - dx) * dy),
                        (x0 + 1, y0 + 1, dx * dy),
                    ):
                        if 0 <= xx < w and 0 <= yy < h:
                            level[yy, xx] += value * factor

            combined += cv2.resize(level, (size, size), interpolation=cv2.INTER_LINEAR)
            start += n_points

        combined = cv2.GaussianBlur(combined, (0, 0), 8)
        combined -= combined.min()
        combined /= combined.max() + 1e-8
        return combined.astype(np.float32)

    def _hook(self, module, args, kwargs, output):
        self._captured = {
            "kwargs": kwargs,
            "weights": output[1]
        }

    def generate(self, image_path, output_path):
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(self.device)

        handle = self.cross_attn.register_forward_hook(self._hook, with_kwargs=True)

        try:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)

            scores, classes = outputs.logits[0].sigmoid().max(dim=-1)
            query_idx = int(scores.argmax().item())

            kw = self._captured["kwargs"]
            hidden = kw["hidden_states"]
            reference = kw["reference_points"]
            spatial_shapes = kw["spatial_shapes"]

            offsets = self.cross_attn.sampling_offsets(hidden).reshape(
                hidden.shape[0],
                hidden.shape[1],
                self.cross_attn.n_heads,
                sum(self.cross_attn.num_points_list),
                2,
            )

            if reference.shape[-1] != 4:
                raise RuntimeError(
                    f"Unsupported DEIMv2 reference shape: {reference.shape}"
                )

            scale = self.cross_attn.num_points_scale.to(hidden.dtype).unsqueeze(-1)

            locations = reference[:, :, None, :, :2] + (
                offsets
                * scale
                * reference[:, :, None, :, 2:]
                * self.cross_attn.offset_scale
            )

            shapes = [
                (int(h), int(w))
                for h, w in spatial_shapes.detach().cpu().tolist()
            ]

            heatmap = self._heatmap(
                locations[0, query_idx].detach().cpu().numpy(),
                self._captured["weights"][0, query_idx].detach().cpu().numpy(),
                shapes,
                list(self.cross_attn.num_points_list),
            )

            resized = np.array(image.resize((640, 640)))
            visualization = show_cam_on_image(
                resized.astype(np.float32) / 255.0,
                heatmap,
                use_rgb=True,
                image_weight=0.65,
            )

            cx, cy, bw, bh = (
                outputs.pred_boxes[0, query_idx].detach().cpu().tolist()
            )

            cx *= 640
            cy *= 640
            bw *= 640
            bh *= 640

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            cv2.rectangle(
                visualization,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            class_idx = int(classes[query_idx].item())
            class_name = self.model.config.id2label.get(
                class_idx,
                str(class_idx),
            )
            confidence = float(scores[query_idx].item())

            cv2.putText(
                visualization,
                f"{class_name} {confidence:.2f}",
                (max(4, x1), max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
            )

            return {
                "output_path": str(output_path),
                "query_index": query_idx,
                "class_id": class_idx,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            }

        finally:
            handle.remove()