# import sys
# sys.path.insert(0, "src")
# from transformers import AutoImageProcessor, Deimv2ForObjectDetection
# from PIL import Image
# import torch
# from pathlib import Path

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )
# processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")

# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]
# image = Image.open(test_img).convert("RGB")
# inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs, output_attentions=True)

# print("Number of cross-attention layers:", len(outputs.cross_attentions))
# print("Shape of last layer's cross-attention:", outputs.cross_attentions[-1].shape)
# print("sqrt of token count:", outputs.cross_attentions[-1].shape[-1] ** 0.5)

# """
# Diagnostic: find the deformable attention module and inspect what it
# actually exposes, before committing to a full rewrite.
# """
# import sys
# sys.path.insert(0, "src")
# from transformers import AutoImageProcessor, Deimv2ForObjectDetection
# from PIL import Image
# import torch
# from pathlib import Path

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )
# processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")

# for name, module in model.named_modules():
#     if "deform" in type(module).__name__.lower():
#         print(name, "->", type(module).__name__)

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# print("Top-level children of model:")
# for name, module in model.named_children():
#     print(" ", name, "->", type(module).__name__)

# print("\nAny module with 'backbone' in its name (no depth limit):")
# for name, module in model.named_modules():
#     if "backbone" in name.lower():
#         print(" ", name, "->", type(module).__name__)

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# print("Children of model.model (the Deimv2Model):")
# for name, module in model.model.named_children():
#     print(" ", name, "->", type(module).__name__)

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# print("Children of model.model.encoder (the Deimv2HybridEncoder):")
# for name, module in model.model.encoder.named_children():
#     print(" ", name, "->", type(module).__name__)

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# for name, module in model.named_modules():
#     depth = name.count(".")
#     if depth <= 2:
#         print(name, "->", type(module).__name__)

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection
# import torch.nn as nn

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# print("First 15 Conv2d layers found anywhere in the model (with full dotted path):")
# count = 0
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(" ", name)
#         count += 1
#         if count >= 15:
#             break

# import sys
# sys.path.insert(0, "src")
# from transformers import Deimv2ForObjectDetection

# model = Deimv2ForObjectDetection.from_pretrained(
#     "results/dino/run/checkpoint-2688", attn_implementation="eager"
# )

# stages = model.model.conv_encoder.model.encoder.stages
# print(f"Number of stages: {len(stages)}")
# print(f"Blocks in last stage: {len(stages[-1].blocks)}")

# import sys
# sys.path.insert(0, "src")
# from explainability.gradcam import generate_yolov8_heatmap
# from ultralytics import YOLO
# from pathlib import Path
# import matplotlib.pyplot as plt

# model = YOLO("results/baseline_yolov8/run/weights/best.pt")
# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]

# # NO bbox this time -- full image, no cropping
# original, overlay, _ = generate_yolov8_heatmap(model, test_img, bbox=None)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(original)
# axes[0].set_title("Original (full image)")
# axes[0].axis("off")
# axes[1].imshow(overlay)
# axes[1].set_title("EigenCAM (full image, no crop)")
# axes[1].axis("off")
# plt.tight_layout()
# plt.savefig("test_full_image_cam.png", dpi=150)
# plt.show()

# import sys
# sys.path.insert(0, "src")
# from ultralytics import YOLO
# from src.explainability.gradcam import generate_yolov8_classspecific_gradcam
# from pathlib import Path
# import matplotlib.pyplot as plt

# model = YOLO("results/baseline_yolov8/run/weights/best.pt")
# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]

# original, overlay, _ = generate_yolov8_classspecific_gradcam(model, test_img)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(original)
# axes[0].set_title("Original")
# axes[0].axis("off")
# axes[1].imshow(overlay)
# axes[1].set_title("YOLOv8n Class-Specific Grad-CAM")
# axes[1].axis("off")
# plt.tight_layout()
# plt.savefig("test_classspecific_yolo.png", dpi=150)
# plt.show()

# import sys
# sys.path.insert(0, "src")
# from ultralytics import RTDETR
# from pathlib import Path
# import cv2
# import torch

# model = RTDETR("results/rtdetr/run/weights/best.pt")
# model.model.eval()

# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]
# img_bgr = cv2.imread(str(test_img))

# if model.predictor is None:
#     model.predict(img_bgr, verbose=False)
# input_tensor = model.predictor.preprocess([img_bgr])

# with torch.no_grad():
#     raw_output = model.model(input_tensor)

# print("Type:", type(raw_output))
# if isinstance(raw_output, (tuple, list)):
#     for i, item in enumerate(raw_output):
#         if hasattr(item, "shape"):
#             print(f"  [{i}] shape: {item.shape}")
#         else:
#             print(f"  [{i}] type: {type(item)}")

# import sys
# sys.path.insert(0, "src")
# from ultralytics import RTDETR
# from pathlib import Path
# import cv2
# import torch

# model = RTDETR("results/rtdetr/run/weights/best.pt")
# model.model.eval()

# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]
# img_bgr = cv2.imread(str(test_img))

# if model.predictor is None:
#     model.predict(img_bgr, verbose=False)
# input_tensor = model.predictor.preprocess([img_bgr])

# with torch.no_grad():
#     raw_output = model.model(input_tensor)

# nested = raw_output[1]
# print("Length of raw_output[1]:", len(nested))
# for i, item in enumerate(nested):
#     if hasattr(item, "shape"):
#         print(f"  [1][{i}] shape: {item.shape}")
#     elif isinstance(item, (tuple, list)):
#         print(f"  [1][{i}] is itself a {type(item).__name__} of length {len(item)}")
#         for j, sub in enumerate(item):
#             if hasattr(sub, "shape"):
#                 print(f"    [1][{i}][{j}] shape: {sub.shape}")
#     else:
#         print(f"  [1][{i}] type: {type(item)}")

# import sys
# sys.path.insert(0, "src")
# from ultralytics import RTDETR
# from explainability.gradcam_classspecific import generate_rtdetr_classspecific_gradcam
# from pathlib import Path
# import matplotlib.pyplot as plt

# model = RTDETR("results/rtdetr/run/weights/best.pt")
# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]

# original, overlay, _ = generate_rtdetr_classspecific_gradcam(model, test_img)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(original)
# axes[0].axis("off")
# axes[1].imshow(overlay)
# axes[1].axis("off")
# plt.tight_layout()
# plt.savefig("test_rtdetr_gradcam.png", dpi=150)
# plt.show()

# import sys
# sys.path.insert(0, "src")
# import json
# from transformers import AutoImageProcessor, Deimv2ForObjectDetection
# from explainability.gradcam_classspecific import generate_dino_classspecific_gradcam
# from pathlib import Path
# import matplotlib.pyplot as plt

# checkpoint_dirs = sorted(Path("results/dino/run").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
# with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
#     best_checkpoint = json.load(f)["best_model_checkpoint"]

# model = Deimv2ForObjectDetection.from_pretrained(best_checkpoint)
# processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")

# test_img = list(Path("data/processed/test/images").glob("*.jpg"))[0]

# original, overlay, _ = generate_dino_classspecific_gradcam(model, processor, test_img)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(original)
# axes[0].axis("off")
# axes[1].imshow(overlay)
# axes[1].axis("off")
# plt.tight_layout()
# plt.savefig("test_dino_gradcam.png", dpi=150)
# plt.show()

# import sys
# sys.path.insert(0, "src")
# import json
# from transformers import AutoImageProcessor, Deimv2ForObjectDetection
# from ultralytics import RTDETR
# from explainability.gradcam_classspecific import generate_dino_classspecific_gradcam, generate_rtdetr_classspecific_gradcam
# from pathlib import Path
# import matplotlib.pyplot as plt

# # Load a real ground-truth box from the test set annotations
# with open("data/processed/test/_annotations.coco.json") as f:
#     coco = json.load(f)

# images_by_id = {img["id"]: img for img in coco["images"]}
# anns_by_image = {}
# for ann in coco["annotations"]:
#     anns_by_image.setdefault(ann["image_id"], []).append(ann)

# # pick the first image that actually has an annotation
# image_id = next(iid for iid, anns in anns_by_image.items() if len(anns) > 0)
# img_info = images_by_id[image_id]
# ann = anns_by_image[image_id][0]
# x, y, w, h = ann["bbox"]
# bbox = (x, y, x + w, y + h)
# img_path = Path("data/processed/test/images") / img_info["file_name"]

# print(f"Using image: {img_info['file_name']}, bbox: {bbox}")

# # --- DEIMv2 ---
# checkpoint_dirs = sorted(Path("results/dino/run").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
# with open(checkpoint_dirs[-1] / "trainer_state.json") as f:
#     best_checkpoint = json.load(f)["best_model_checkpoint"]
# dino_model = Deimv2ForObjectDetection.from_pretrained(best_checkpoint)
# dino_processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")

# dino_original, dino_overlay, _ = generate_dino_classspecific_gradcam(dino_model, dino_processor, img_path, bbox=bbox)

# # --- RT-DETR ---
# rtdetr_model = RTDETR("results/rtdetr/run/weights/best.pt")
# rtdetr_original, rtdetr_overlay, _ = generate_rtdetr_classspecific_gradcam(rtdetr_model, img_path, bbox=bbox)

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes[0, 0].imshow(dino_original); axes[0, 0].set_title("Original (cropped to GT bbox)"); axes[0, 0].axis("off")
# axes[0, 1].imshow(dino_overlay); axes[0, 1].set_title("DEIMv2 Grad-CAM (with bbox crop)"); axes[0, 1].axis("off")
# axes[1, 0].imshow(rtdetr_original); axes[1, 0].set_title("Original (cropped to GT bbox)"); axes[1, 0].axis("off")
# axes[1, 1].imshow(rtdetr_overlay); axes[1, 1].set_title("RT-DETR Grad-CAM (with bbox crop)"); axes[1, 1].axis("off")
# plt.tight_layout()
# plt.savefig("test_with_bbox_crop.png", dpi=150)
# plt.show()

import sys
sys.path.insert(0, "src")
from ultralytics import RTDETR
from explainability.gradcam_classspecific import generate_rtdetr_classspecific_gradcam
from pathlib import Path
import matplotlib.pyplot as plt
import json

with open("data/processed/test/_annotations.coco.json") as f:
    coco = json.load(f)

images_by_id = {img["id"]: img for img in coco["images"]}
anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

image_id = next(iid for iid, anns in anns_by_image.items() if len(anns) > 0)
img_info = images_by_id[image_id]
ann = anns_by_image[image_id][0]
x, y, w, h = ann["bbox"]
bbox = (x, y, x + w, y + h)
img_path = Path("data/processed/test/images") / img_info["file_name"]

model = RTDETR("results/rtdetr/run/weights/best.pt")
original, overlay, _ = generate_rtdetr_classspecific_gradcam(model, img_path, bbox=bbox)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original); axes[0].axis("off")
axes[1].imshow(overlay); axes[1].axis("off")
plt.tight_layout()
plt.savefig("rtdetr_confirmed_result.png", dpi=150)
plt.show()