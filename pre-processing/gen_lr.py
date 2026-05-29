import cv2
import numpy as np
import os
from tqdm import tqdm

def degrade_image(img):
    # img = BGR, 256x256

    # Step 1: Downscale 256 → 85
    lr = cv2.resize(
        img,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    # Step 2: Upscale 85 → 256 (bicubic)
    lr = cv2.resize(
        lr,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    # Step 3: Gaussian Blur
    lr = cv2.GaussianBlur(
        lr, (0, 0), sigmaX=1.5
    )

    # Step 4: Gaussian Noise
    noise = np.random.normal(
        0, 4, lr.shape
    ).astype(np.float32)

    lr = np.clip(
        lr.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    return lr

def generate_lr_dataset(
    input_dir,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    classes = sorted([
        c for c in os.listdir(input_dir)
        if os.path.isdir(
            os.path.join(input_dir, c)
        )
    ])

    print(f"Classes: {classes}")
    total = 0

    for cls in classes:
        in_cls  = os.path.join(input_dir, cls)
        out_cls = os.path.join(output_dir, cls)
        os.makedirs(out_cls, exist_ok=True)

        images = [
            f for f in os.listdir(in_cls)
            if f.lower().endswith(
                ('.jpg', '.jpeg', '.png')
            )
        ]

        for img_name in tqdm(
            images, desc=cls
        ):
            img = cv2.imread(
                os.path.join(in_cls, img_name)
            )
            if img is None:
                continue

            # Resize to 256x256 first
            img = cv2.resize(img, (256, 256))

            # Degrade
            lr = degrade_image(img)

            # Save
            cv2.imwrite(
                os.path.join(out_cls, img_name),
                lr
            )
            total += 1

    print(f"\n{output_dir}: {total} images ✅")

# ─────────────────────────────
# Generate Both Datasets
# ─────────────────────────────

print("=== LR_Training ===")
generate_lr_dataset(
    "dataset/Training",
    "dataset/LOW_Training"
)

print("\n=== LR_Testing ===")
generate_lr_dataset(
    "dataset/Testing",
    "dataset/LOW_Testing"
)