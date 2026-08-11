import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMG_SIZE = 300
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = './models/effnetb3.pth'
BASE_DIR = './dataset'
OUTPUT_DIR = './gradcam_results'

DATASET_FOLDERS = {
    'HR': 'Testing',
    'LR': 'LR_Testing',
    'SRCNN': 'SRCNN_Testing',
    'EDSR': 'EDSR_Testing',
}

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model(num_classes=4):
    m = models.efficientnet_b3(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_feats, num_classes))
    return m.to(device)

classifier = build_model(num_classes=4)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
classifier.eval()
print("Classifier loaded")

target_layers = [classifier.features[-1]]

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

def save_single_gradcam(img_path, true_class, out_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = test_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    cam = GradCAM(model=classifier, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = denormalize(input_tensor[0])
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(grayscale_cam, cmap="jet"); axes[1].set_title("Heatmap"); axes[1].axis("off")
    axes[2].imshow(visualization); axes[2].set_title("Overlay"); axes[2].axis("off")

    title = f"Pred: {CLASSES[pred_class]} ({confidence*100:.1f}%) | True: {true_class}"
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return pred_class, confidence

if __name__ == "__main__":
    for dataset_label, folder_name in DATASET_FOLDERS.items():
        out_folder = os.path.join(OUTPUT_DIR, dataset_label)
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n--- Processing {dataset_label} ({folder_name}) ---")

        for cls_name in CLASSES:
            cls_dir = os.path.join(BASE_DIR, folder_name, cls_name)
            if not os.path.exists(cls_dir):
                print(f"  SKIP: {cls_dir} not found")
                continue
            files = sorted(os.listdir(cls_dir))
            sample_file = files[0]  # first image per class - change index for a different sample
            img_path = os.path.join(cls_dir, sample_file)
            out_path = os.path.join(out_folder, f"{cls_name}_gradcam.png")

            pred, conf = save_single_gradcam(img_path, cls_name, out_path)
            print(f"  {cls_name} ({sample_file}): pred={CLASSES[pred]} ({conf*100:.1f}%) -> saved {out_path}")

    print("\nDone. Folders created:")
    for label in DATASET_FOLDERS:
        print(f"  {OUTPUT_DIR}/{label}/")