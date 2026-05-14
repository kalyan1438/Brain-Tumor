import os
import cv2

input_dir = "/home/kalyan1438/Brain-Tumor/dataset/Testing"
output_dir = "/home/kalyan1438/Brain-Tumor/dataset/LR_Testing"

classes = os.listdir(input_dir)

for cls in classes:

    input_class = os.path.join(input_dir, cls)
    output_class = os.path.join(output_dir, cls)

    os.makedirs(output_class, exist_ok=True)

    for img_name in os.listdir(input_class):
        print("Processing:", img_name)
        img_path = os.path.join(input_class, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Original size
        h, w = img.shape[:2]

        # Downsample
        small = cv2.resize(img, (w//2, h//2))

        # Upsample back
        lr = cv2.resize(small, (w, h))

        # Mild blur
        lr = cv2.GaussianBlur(lr, (3,3), 0)

        save_path = os.path.join(output_class, img_name)

        cv2.imwrite(save_path, lr)

print("LR dataset generated successfully")