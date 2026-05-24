import os
import cv2
import numpy as np
from tqdm import tqdm

# ========= PATHS =========

INPUT_DIR = "../dataset/Training"
OUTPUT_DIR = "../dataset/LR_Training" 
INPUT_DIR_ = "../dataset/Testing"
OUTPUT_DIR_ = "../dataset/LR_Testing" 
# ========= CREATE OUTPUT =========

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= PROCESS =========

for cls in os.listdir(INPUT_DIR):

    input_cls = os.path.join(INPUT_DIR, cls)
    output_cls = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(output_cls, exist_ok=True)

    images = os.listdir(input_cls)

    print(f"\nProcessing {cls}...")

    for img_name in tqdm(images):

        img_path = os.path.join(input_cls, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # ==================================
        # STEP 1: Downscale (224 -> 112)
        # ==================================

        h, w = img.shape[:2]

        lr = cv2.resize(
            img,
            (w // 2, h // 2),
            interpolation=cv2.INTER_AREA
        )

        # ==================================
        # STEP 2: Upscale back
        # ==================================

        lr = cv2.resize(
            lr,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        # ==================================
        # STEP 3: Mild Gaussian Blur
        # ==================================

        lr = cv2.GaussianBlur(
            lr,
            (3, 3),
            1.25
        )

        # ==================================
        # STEP 4: Low Gaussian Noise
        # ==================================

        noise = np.random.normal(
            0,
            7,
            lr.shape
        ).astype(np.float32)

        lr = lr.astype(np.float32) + noise

        lr = np.clip(lr, 0, 255).astype(np.uint8)

        # ==================================
        # SAVE
        # ==================================

        save_path = os.path.join(
            output_cls,
            img_name
        )

        cv2.imwrite(save_path, lr)

print("\nLR dataset generation completed ✅")