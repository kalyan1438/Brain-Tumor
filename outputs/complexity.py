# ============================================================
# Model Complexity Analysis
# SRCNN + EDSR + EfficientNet-B3
# ============================================================

import os
import time
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("MODEL COMPLEXITY ANALYSIS")
print("=" * 75)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. CHECKPOINT PATHS
# CHANGE THESE TO YOUR ACTUAL FILE LOCATIONS
# ============================================================

SRCNN_PATH = "../models/srcnn_mri.pth"
EDSR_PATH = "../models/edsr_mri.pth"
EFFNET_PATH = "../models/effnetb3.pth"


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\nChecking checkpoints...")

for path in [SRCNN_PATH, EDSR_PATH, EFFNET_PATH]:

    if os.path.exists(path):
        print("FOUND :", path)
    else:
        print("MISSING:", path)


# ============================================================
# 4. EXACT SRCNN ARCHITECTURE
# From your training code
# ============================================================

class SRCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=9,
            padding=4
        )

        self.conv2 = nn.Conv2d(
            64,
            32,
            kernel_size=1,
            padding=0
        )

        self.conv3 = nn.Conv2d(
            32,
            3,
            kernel_size=5,
            padding=2
        )

    def forward(self, x):

        x = F.relu(
            self.conv1(x)
        )

        x = F.relu(
            self.conv2(x)
        )

        x = self.conv3(x)

        return x


# ============================================================
# 5. EXACT EDSR ARCHITECTURE
# From your training code
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        n_feats=64
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.block(x)


class EDSR(nn.Module):

    def __init__(
        self,
        n_resblocks=16,
        n_feats=64
    ):

        super().__init__()

        self.input_conv = nn.Conv2d(
            3,
            n_feats,
            kernel_size=3,
            padding=1
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(n_feats)
                for _ in range(n_resblocks)
            ]
        )

        self.middle_conv = nn.Conv2d(
            n_feats,
            n_feats,
            kernel_size=3,
            padding=1
        )

        self.output_conv = nn.Conv2d(
            n_feats,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x1 = self.input_conv(x)

        res = self.res_blocks(x1)

        res = self.middle_conv(res)

        x2 = x1 + res

        out = self.output_conv(x2)

        return out


# ============================================================
# 6. EXACT EFFICIENTNET-B3 CLASSIFIER
# From your training code
# ============================================================

def build_effnet_b3():

    model = models.efficientnet_b3(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(
            in_features,
            4
        )
    )

    return model


# ============================================================
# 7. LOAD CHECKPOINT
# ============================================================

def load_model_weights(
    model,
    path,
    model_name
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{model_name} checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device
    )

    # --------------------------------------------------------
    # Case 1: direct state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise ValueError(
            f"Unsupported checkpoint format for {model_name}"
        )

    # --------------------------------------------------------
    # Remove DataParallel "module." prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError as e:

        print(
            f"\nERROR loading {model_name}"
        )

        print(e)

        raise

    print(
        f"{model_name} weights loaded successfully."
    )

    return model


# ============================================================
# 8. CREATE + LOAD MODELS
# ============================================================

print("\n" + "=" * 75)
print("LOADING MODELS")
print("=" * 75)


# -----------------------
# SRCNN
# -----------------------

srcnn = SRCNN().to(device)

srcnn = load_model_weights(
    srcnn,
    SRCNN_PATH,
    "SRCNN"
)

srcnn.eval()


# -----------------------
# EDSR
# -----------------------

edsr = EDSR(
    n_resblocks=16,
    n_feats=64
).to(device)

edsr = load_model_weights(
    edsr,
    EDSR_PATH,
    "EDSR"
)

edsr.eval()


# -----------------------
# EfficientNet-B3
# -----------------------

classifier = build_effnet_b3().to(device)

classifier = load_model_weights(
    classifier,
    EFFNET_PATH,
    "EfficientNet-B3"
)

classifier.eval()


# ============================================================
# 9. SANITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("SANITY CHECK")
print("=" * 75)


with torch.inference_mode():

    # SRCNN
    x_srcnn = torch.randn(
        1, 3, 256, 256,
        device=device
    )

    y_srcnn = srcnn(x_srcnn)

    print(
        "SRCNN:",
        x_srcnn.shape,
        "->",
        y_srcnn.shape
    )

    # EDSR
    x_edsr = torch.randn(
        1, 3, 256, 256,
        device=device
    )

    y_edsr = edsr(x_edsr)

    print(
        "EDSR:",
        x_edsr.shape,
        "->",
        y_edsr.shape
    )

    # EfficientNet
    x_effnet = torch.randn(
        1, 3, 300, 300,
        device=device
    )

    y_effnet = classifier(x_effnet)

    print(
        "EfficientNet-B3:",
        x_effnet.shape,
        "->",
        y_effnet.shape
    )


# ============================================================
# 10. MODEL STATISTICS
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def compute_model_size_mb(model):

    # FP32 approximation
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    return (
        total_params * 4
        / (1024 ** 2)
    )


# ============================================================
# 11. INFERENCE TIMING
# ============================================================

def measure_inference_time(
    model,
    input_shape,
    warmup_runs=10,
    benchmark_runs=50
):

    dummy_input = torch.randn(
        *input_shape,
        device=device
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    with torch.inference_mode():

        for _ in range(warmup_runs):

            _ = model(
                dummy_input
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    times = []

    with torch.inference_mode():

        for _ in range(benchmark_runs):

            if device.type == "cuda":

                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(
                dummy_input
            )

            if device.type == "cuda":

                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append(
                end - start
            )

    times = np.array(times) * 1000

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times))
    }


# ============================================================
# 12. ANALYZE ONE MODEL
# ============================================================

def analyze_model(
    model,
    model_name,
    input_shape
):

    total_params, trainable_params = count_parameters(
        model
    )

    size_mb = compute_model_size_mb(
        model
    )

    timing = measure_inference_time(
        model,
        input_shape
    )

    result = {

        "Model": model_name,

        "Input": str(
            input_shape[1:]
        ),

        "Parameters": total_params,

        "Trainable": trainable_params,

        "Size_MB": size_mb,

        "Inference_ms": timing["mean"],

        "Std_ms": timing["std"],

        "Min_ms": timing["min"],

        "Max_ms": timing["max"]
    }

    return result


# ============================================================
# 13. RUN ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("RUNNING COMPLEXITY ANALYSIS")
print("=" * 75)


results = []


# SRCNN
results.append(
    analyze_model(
        srcnn,
        "SRCNN",
        (
            1,
            3,
            256,
            256
        )
    )
)


# EDSR
results.append(
    analyze_model(
        edsr,
        "EDSR",
        (
            1,
            3,
            256,
            256
        )
    )
)


# EfficientNet-B3
results.append(
    analyze_model(
        classifier,
        "EfficientNet-B3",
        (
            1,
            3,
            300,
            300
        )
    )
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 115)

print(
    f"{'Model':<20}"
    f"{'Input':<18}"
    f"{'Parameters':<18}"
    f"{'Size(MB)':<12}"
    f"{'Avg(ms)':<12}"
    f"{'Std(ms)':<12}"
)

print("=" * 115)

for r in results:

    print(
        f"{r['Model']:<20}"
        f"{r['Input']:<18}"
        f"{r['Parameters']:<18,}"
        f"{r['Size_MB']:<12.2f}"
        f"{r['Inference_ms']:<12.2f}"
        f"{r['Std_ms']:<12.2f}"
    )

print("=" * 115)


# ============================================================
# 15. SAVE CSV
# ============================================================

os.makedirs(
    "./outputs",
    exist_ok=True
)

csv_path = "./outputs/model_complexity.csv"

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Input",
            "Parameters",
            "Trainable",
            "Size_MB",
            "Inference_ms",
            "Std_ms",
            "Min_ms",
            "Max_ms"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 16. FINISH
# ============================================================

print(
    f"\nResults saved to:\n{csv_path}"
)

print("\nAnalysis completed successfully.")