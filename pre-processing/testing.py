import os
import cv2
import random
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim
)

# Reproducibility
random.seed(42)
np.random.seed(42)

def quick_check(hr_folder, lr_folder):

    psnr_list = []
    ssim_list = []

    for cls in os.listdir(hr_folder):

        hr_cls = os.path.join(hr_folder, cls)
        lr_cls = os.path.join(lr_folder, cls)

        if not os.path.isdir(hr_cls):
            continue

        all_images = [
            f for f in os.listdir(hr_cls)
            if f.lower().endswith(
                ('.jpg', '.jpeg', '.png')
            )
        ]

        sample_size = min(20, len(all_images))

        images = random.sample(
            all_images,
            sample_size
        )

        for img_name in images:

            hr_path = os.path.join(
                hr_cls,
                img_name
            )

            lr_path = os.path.join(
                lr_cls,
                img_name
            )

            hr = cv2.imread(
                hr_path,
                cv2.IMREAD_GRAYSCALE
            )

            lr = cv2.imread(
                lr_path,
                cv2.IMREAD_GRAYSCALE
            )

            if hr is None or lr is None:
                continue

            hr = cv2.resize(
                hr,
                (256, 256)
            )

            lr = cv2.resize(
                lr,
                (256, 256)
            )

            psnr_list.append(
                psnr(hr, lr)
            )

            ssim_list.append(
                ssim(hr, lr)
            )

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n=== Degradation Check ===")
    print(f"LR PSNR: {avg_psnr:.2f} dB")
    print(f"LR SSIM: {avg_ssim:.3f}")

    if 28 <= avg_psnr <= 32:

        print("✅ Perfect degradation!")

    elif avg_psnr < 28:

        print("⚠️ Too strong — reduce blur/noise")

    else:

        print("⚠️ Too mild — increase blur/noise")

    return avg_psnr, avg_ssim


# Run Check
quick_check(
    "dataset/Testing",
    "dataset/LR_Testing"
)