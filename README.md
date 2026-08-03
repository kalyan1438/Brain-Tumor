# Super-Resolution Enhanced Brain Tumor Classification Using EfficientNetB3 Under Degraded MRI Conditions

## Overview

This project investigates how image degradation affects deep learning–based brain tumor classification accuracy, and how much of that lost accuracy can be recovered using super-resolution (SR) techniques. A classifier trained on high-resolution (HR) MRI scans is evaluated across four progressively enhanced versions of degraded test images: raw low-resolution (LR), SRCNN-restored, and EDSR-restored — quantifying the diagnostic value of SR as a pre-processing step.

## Pipeline

```
HR MRI  ──► Degradation (simulated) ──► LR MRI
LR MRI  ──► SRCNN  ──► SR (SRCNN) MRI
LR MRI  ──► EDSR   ──► SR (EDSR) MRI

[ HR / LR / SRCNN / EDSR ] ──► EfficientNetB3 Classifier ──► Tumor Type (4-class)
                                        │
                                        └──► Grad-CAM (explainability)
```

## Dataset

- Source: Kaggle Brain Tumor MRI Dataset
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Training: 5,600 images
- Testing: 1,592 images

## Results

### Classification Accuracy by Degradation Severity

| Degradation Severity | HR     | LR     | SRCNN  | EDSR   |
|-----------------------|--------|--------|--------|--------|
| **Severe**            | 93.72% | 35.18% | 64.51% | 77.39% |
| **Moderate**          | 93.72% | 71.23% | 84.74% | 90.95% |

### Super-Resolution Quality Metrics (Moderate degradation)

| Model  | PSNR     | SSIM   |
|--------|----------|--------|
| SRCNN  | 36.56 dB | 0.971  |
| EDSR   | 38.61 dB | 0.979  |

### Classifier Performance (HR-trained EfficientNetB3)

- Overall test accuracy: **93.72%**
- Strong performance on meningioma, notumor, and pituitary classes
- Glioma is the primary source of misclassification (recall ≈ 0.80), largely due to visual overlap with meningioma in certain scans

## Explainability (Grad-CAM)

- Correct predictions show tight, anatomically consistent attention on the tumor region
- Misclassified examples show attention on plausible but ambiguous regions rather than random/background noise, indicating genuine diagnostic difficulty rather than model failure
- A minor recurring low-intensity activation was observed in the image corner across multiple samples — noted as a limitation for future investigation

## Key Finding

Super-resolution meaningfully recovers classification accuracy lost to image degradation, with EDSR consistently outperforming SRCNN across both degradation severities. This suggests SR pre-processing has practical value in improving diagnostic reliability when working with lower-quality or compressed MRI scans (e.g., legacy hardware, telemedicine transmission).

## Project Structure

```
├── dataset/                # Training/Testing HR + LR + SR image folders
├── models/                 # Trained model weights (.pth)
├── classification/         # EfficientNetB3 training + evaluation scripts
├── pre-processing/         # Degradation + SR generation scripts
├── notebooks/               # Colab notebooks (SRCNN.ipynb, EDSR.ipynb, etc.)
├── gradcam_results/         # Grad-CAM visualization outputs
├── outputs/                 # Result tables, plots, logs
└── paper/                   # Draft manuscript and figures
```

## Models

- **SRCNN**: 3-layer convolutional super-resolution network (Dong et al., 2014 architecture)
- **EDSR**: 16-block residual network without upsampling (same-size refinement variant, Lim et al., 2017 architecture)
- **EfficientNetB3**: ImageNet-pretrained, fine-tuned in two phases (frozen backbone → full fine-tune) with class-weighted loss to address glioma class imbalance

## Limitations

- Single train/test split (no cross-validation reported yet)
- Corner-region activation artifact observed in a subset of Grad-CAM visualizations, requiring further investigation
- Degradation parameters are simulated approximations of real-world MRI quality loss, not derived from actual scanner hardware variation
