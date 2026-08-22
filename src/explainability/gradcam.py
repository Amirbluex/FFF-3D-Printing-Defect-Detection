"""
gradcam.py

Phase 8: Explainability for the CNN model (YOLOv8n).

Uses EigenCAM (Muhammad & Yeasin, 2020) rather than classic Grad-CAM:
EigenCAM doesn't require backpropagating through a class-specific
score (which YOLO's combined detection+classification head makes
awkward to define cleanly) -- instead it takes the first principal
component of a chosen layer's activation maps via SVD, which has been
shown to highlight salient regions comparably to Grad-CAM for
detection models. This is a self-contained implementation (not
dependent on cloning an external, non-pip-installable package).

IMPORTANT: EigenCAM is unsupervised and class-agnostic -- it
highlights whatever varies most strongly across the WHOLE input
image, with no awareness of which object was actually detected. On
this dataset, the printer's fan/nozzle assembly is a high-contrast,
mechanically complex structure present in every image, and initially
dominated the heatmap regardless of where the actual defect was. The
fix: crop tightly around the detected/ground-truth bounding box
(with padding) BEFORE running EigenCAM, so irrelevant high-contrast
structures elsewhere in the frame can't compete for the principal
component.
"""

import cv2
import numpy as np
import torch

def _letterbox(img, target_size):
    """
    Pads img to a square canvas (gray fill) preserving aspect ratio,
    then resizes to target_size x target_size -- matching YOLO's own
    standard preprocessing, instead of naively stretching with
    cv2.resize (which distorts thin/wide crops badly and can produce
    resampling artifacts that dominate EigenCAM's unsupervised
    computation).

    Returns: (letterboxed_img, scale, pad_x, pad_y) -- scale/pad
    needed to map results back to the original crop's coordinates.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # gray fill, YOLO's standard letterbox color
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def _unletterbox_cam(cam, scale, pad_x, pad_y, orig_h, orig_w):
    """
    Reverses the letterbox transform on a CAM computed on the padded
    square, mapping it back to the original (pre-padding) crop shape.
    """
    target_size = cam.shape[0]
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    cam_cropped = cam[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    cam_original_size = cv2.resize(cam_cropped, (orig_w, orig_h))

    return cam_original_size


class EigenCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def __call__(self, input_tensor):
        with torch.no_grad():
            self.model(input_tensor)

        activations = self.activations[0]  # (C, H, W)
        C, H, W = activations.shape
        flat = activations.reshape(C, H * W).cpu().numpy()

        # First principal component via SVD, projected back onto the
        # spatial grid -- this is EigenCAM's core computation.
        flat = flat - flat.mean(axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        cam = Vt[0].reshape(H, W)

        cam = np.maximum(cam, 0)  # keep only positive activation
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hook(self):
        self.hook.remove()


def _crop_to_bbox(img_rgb, bbox, padding_ratio):
    """
    Crops img_rgb to bbox (x1, y1, x2, y2) with padding. Returns the
    full image unchanged if bbox is None.
    """
    if bbox is None:
        return img_rgb

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding_ratio, h * padding_ratio
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(img_rgb.shape[1], int(x2 + pad_x))
    crop_y2 = min(img_rgb.shape[0], int(y2 + pad_y))

    return img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]


def generate_yolov8_heatmap(model, img_path, target_layer_index=-2, imgsz=640,
                             bbox=None, padding_ratio=0.6):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = _crop_to_bbox(img_rgb, bbox, padding_ratio)

    letterboxed, scale, pad_x, pad_y = _letterbox(display_img, imgsz)

    input_tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    device = next(model.model.parameters()).device
    input_tensor = input_tensor.to(device)

    target_layer = model.model.model[target_layer_index]
    cam_extractor = EigenCAM(model.model, target_layer)
    grayscale_cam = cam_extractor(input_tensor)
    cam_extractor.remove_hook()

    # grayscale_cam comes out at the target layer's own spatial resolution
    # (smaller than imgsz) -- resize up to imgsz first, THEN unletterbox
    grayscale_cam_full = cv2.resize(grayscale_cam, (imgsz, imgsz))
    grayscale_cam_resized = _unletterbox_cam(
        grayscale_cam_full, scale, pad_x, pad_y, display_img.shape[0], display_img.shape[1]
    )

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap_rgb, 0.4, 0)

    return display_img, overlay, grayscale_cam_resized