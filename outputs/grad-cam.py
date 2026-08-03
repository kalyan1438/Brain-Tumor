import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================================
# Configuration
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

IMG_SIZE = 300

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

MODEL_PATH = "./models/effnetb3_final.pth"
TESTING_DIR = "./dataset/Testing"

OUTPUT_DIR = "./gradcam_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES_PER_CLASS = 3
NUM_MISCLASSIFIED = 5

random.seed(42)

# ==========================================================
# Image Transform
# ==========================================================

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# Model
# ==========================================================

def build_model(num_classes=4):

    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model.to(device)

model = build_model(4)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# Automatically find last Conv layer
# ==========================================================

def get_last_conv_layer(model):

    last_conv = None
    last_name = ""

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            last_conv = module
            last_name = name

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found.")

    print("GradCAM Layer :", last_name)

    return last_conv

target_layers = [get_last_conv_layer(model)]

# ==========================================================
# Utility
# ==========================================================

def denormalize(img_tensor):

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    img = img_tensor.cpu()*std + mean

    return img.clamp(0,1).permute(1,2,0).numpy()

# ==========================================================
# GradCAM
# ==========================================================

def show_gradcam(img_path,true_class=None,save_path=None):

    img = Image.open(img_path).convert("RGB")

    input_tensor = test_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        probs = torch.softmax(output,dim=1)

        confidence,pred = probs.max(dim=1)

        pred = pred.item()
        confidence = confidence.item()

    cam = HiResCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pred)],
        aug_smooth=True,
        eigen_smooth=True
    )[0]

    rgb_img = denormalize(input_tensor[0])

    visualization = show_cam_on_image(
        rgb_img.astype("float32"),
        grayscale_cam,
        use_rgb=True,
        image_weight=0.6
    )

    fig,axes = plt.subplots(1,3,figsize=(13,4))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam,cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = f"Prediction: {CLASSES[pred]} ({confidence*100:.2f}%)"

    if true_class is not None:
        title += f" | True: {true_class}"

    fig.suptitle(title,fontsize=16)

    plt.tight_layout()

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()

    print("Saved ->",save_path)

    return pred,confidence

# ==========================================================
# Find Misclassified Images
# ==========================================================

def find_misclassified(max_images=5):

    misclassified=[]

    for cls in CLASSES:

        folder=os.path.join(TESTING_DIR,cls)

        for fname in os.listdir(folder):

            img_path=os.path.join(folder,fname)

            img=Image.open(img_path).convert("RGB")

            tensor=test_tfms(img).unsqueeze(0).to(device)

            with torch.no_grad():

                pred=model(tensor).argmax(1).item()

            if CLASSES[pred]!=cls:

                misclassified.append(
                    (
                        img_path,
                        cls,
                        CLASSES[pred]
                    )
                )

                if len(misclassified)>=max_images:
                    return misclassified

    return misclassified

# ==========================================================
# Main
# ==========================================================

if __name__=="__main__":

    print("\nGenerating GradCAM Images...\n")

    for cls in CLASSES:

        folder=os.path.join(TESTING_DIR,cls)

        files=random.sample(
            os.listdir(folder),
            min(NUM_SAMPLES_PER_CLASS,len(os.listdir(folder)))
        )

        for i,f in enumerate(files):

            path=os.path.join(folder,f)

            show_gradcam(
                path,
                true_class=cls,
                save_path=os.path.join(
                    OUTPUT_DIR,
                    f"{cls}_{i+1}.png"
                )
            )

    print("\nSearching Misclassified Images...\n")

    mistakes=find_misclassified(NUM_MISCLASSIFIED)

    if len(mistakes)==0:

        print("No Misclassified Images Found.")

    else:

        for i,(img_path,true_cls,pred_cls) in enumerate(mistakes):

            print(
                f"{i+1}. True={true_cls} Pred={pred_cls}"
            )

            show_gradcam(
                img_path,
                true_class=true_cls,
                save_path=os.path.join(
                    OUTPUT_DIR,
                    f"misclassified_{i+1}.png"
                )
            )

    print("\nFinished Successfully!")