"""
attention_rollout.py

Phase 8: Explainability for the transformer detectors.

INVESTIGATION FINDING: classic Attention Rollout is architecturally
inapplicable to RT-DETR and DEIMv2 (and, by extension, real DINO,
which shares the same lineage). All three use multiscale deformable
attention (confirmed via module inspection:
Deimv2MultiscaleDeformableAttention), which outputs sparse sampling
points + weights (shape: batch, queries, heads, sampling_points) --
NOT a dense attention matrix. output_attentions=True exposes the
weights but not the sampling_locations needed to place them
spatially, and that tensor isn't exposed through the public API.

Both RT-DETR-L and DEIMv2-Nano therefore use the SAME EigenCAM
technique as YOLOv8n (see gradcam.py), each targeting their own
architecture's backbone final layer. This is documented explicitly
as a substitution driven by a real architectural constraint, not an
arbitrary choice -- and is a legitimate, citable finding about this
model family, not a limitation of this project's pipeline.

Both functions crop tightly to the detected/ground-truth bounding box
before running EigenCAM, for the same reason described in
gradcam.py's docstring (unrelated high-contrast structures elsewhere
in the frame otherwise dominate the unsupervised CAM computation).
"""

import cv2
import numpy as np
import torch
from PIL import Image

from .gradcam import EigenCAM, _crop_to_bbox, _letterbox, _unletterbox_cam


def generate_rtdetr_heatmap_eigencam_fallback(model, img_path, imgsz=640,
                                               bbox=None, padding_ratio=0.6):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = _crop_to_bbox(img_rgb, bbox, padding_ratio)

    letterboxed, scale, pad_x, pad_y = _letterbox(display_img, imgsz)

    input_tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    device = next(model.model.parameters()).device
    input_tensor = input_tensor.to(device)

    target_layer = model.model.model[-5]
    cam_extractor = EigenCAM(model.model, target_layer)
    grayscale_cam = cam_extractor(input_tensor)
    cam_extractor.remove_hook()

    grayscale_cam_full = cv2.resize(grayscale_cam, (imgsz, imgsz))
    grayscale_cam_resized = _unletterbox_cam(
        grayscale_cam_full, scale, pad_x, pad_y, display_img.shape[0], display_img.shape[1]
    )

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap_rgb, 0.4, 0)

    return display_img, overlay, grayscale_cam_resized


def generate_dino_heatmap_eigencam(model, image_processor, img_path,
                                    bbox=None, padding_ratio=0.6):
    image_pil = Image.open(img_path).convert("RGB")
    img_rgb_full = np.array(image_pil)
    display_img = _crop_to_bbox(img_rgb_full, bbox, padding_ratio)

    letterboxed, scale, pad_x, pad_y = _letterbox(display_img, 640)  # DEIMv2's processor will further resize internally, but starting from a proper letterbox avoids feeding it a badly-stretched crop

    cropped_pil = Image.fromarray(letterboxed)
    inputs = image_processor(images=cropped_pil, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    target_layer = model.model.conv_encoder.model.encoder.stages[-1].blocks[-1]
    cam_extractor = EigenCAM(model, target_layer)

    with torch.no_grad():
        model(**inputs)
    activations = cam_extractor.activations[0]
    cam_extractor.remove_hook()

    C, H, W = activations.shape
    flat = activations.reshape(C, H * W).cpu().numpy()
    flat = flat - flat.mean(axis=1, keepdims=True)

    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    cam = Vt[0].reshape(H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_full = cv2.resize(cam, (640, 640))
    cam_resized = _unletterbox_cam(cam_full, scale, pad_x, pad_y, display_img.shape[0], display_img.shape[1])

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap_rgb, 0.4, 0)

    return display_img, overlay, cam_resized