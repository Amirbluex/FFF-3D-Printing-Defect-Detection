import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sample_filename = list((PROJECT_ROOT / "data" / "processed" / "test" / "images").glob("*.jpg"))[0].name

corruptions_to_check = ["lighting_darker", "noise_gaussian", "blur_motion", "compression_jpeg20"]

# fig, axes = plt.subplots(1, 5, figsize=(20, 4))

original = cv2.cvtColor(cv2.imread(str(PROJECT_ROOT / "data" / "processed" / "test" / "images" / sample_filename)), cv2.COLOR_BGR2RGB)
# axes[0].imshow(original)
# axes[0].set_title("Original (clean)")
# axes[0].axis("off")

# for i, corruption in enumerate(corruptions_to_check):
#     img_path = PROJECT_ROOT / "data" / "corrupted" / corruption / "images" / sample_filename
#     img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
#     axes[i+1].imshow(img)
#     axes[i+1].set_title(corruption)
#     axes[i+1].axis("off")

# plt.tight_layout()
# plt.show()

import numpy as np

original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)

for corruption in ["lighting_darker", "noise_gaussian", "blur_motion", "compression_jpeg20"]:
    img_path = PROJECT_ROOT / "data" / "corrupted" / corruption / "images" / sample_filename
    corrupted_img = cv2.imread(str(img_path))
    corrupted_gray = cv2.cvtColor(corrupted_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean_abs_diff = np.mean(np.abs(original_gray - corrupted_gray))
    print(f"{corruption}: mean absolute pixel difference = {mean_abs_diff:.2f}")

print(f"\nOriginal image resolution: {original.shape}")