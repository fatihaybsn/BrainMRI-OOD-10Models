<p align="right">🇬🇧 English | <a href="README-TR.md">🇹🇷 Türkçe</a></p>

# Brain MRI Tumor vs No-Tumor — OOD Generalization (10 Models)

This repository presents my work on binary **Tumor / No Tumor** classification from brain MRI images under an **external test (OOD) scenario with a real-world-like resolution distribution**.

## 🌐 Portfolio Project Page

For a more visual and detailed presentation of the project, architecture comparisons, and highlighted results, visit the project page on my portfolio website:

> **[View the BrainMRI OOD — 10 Models Project Page](https://fathaybasn.me/projects/brainmri-ood-10models)**

## 📊 Detailed Results & Visualizations

The table in this README is a summary. For all comparative graphs across the 10 models (Accuracy/Loss curves and ROC curves), error analyses, and training parameters, see the comprehensive technical report:

👉 **[Download / View Technical Report (PDF)](./reports/Brain_Tumor_Classification_Report.pdf)**


> ⚠️ **Not for medical use.** This work is intended for educational and research purposes only.

## 🤗 Trained Models — Hugging Face

[![Hugging Face Model Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-10%20Architectures%20%7C%2013%20Checkpoints-yellow)](https://huggingface.co/collections/Fatihaybasn/brainmri-ood-benchmark-10-architectures-13-checkpoints)

The **13 checkpoints from 10 architectures** trained in this project have been published on Hugging Face together with model cards, OOD metrics, available result figures, loading code, and SHA-256 records:

> **[BrainMRI OOD Benchmark — 10 Architectures, 13 Checkpoints](https://huggingface.co/collections/Fatihaybasn/brainmri-ood-benchmark-10-architectures-13-checkpoints)**

Weights for the best experiment: **[Custom MSAF + EfficientNet-B0 — Augmentation 0.3](https://huggingface.co/Fatihaybasn/brainmri-ood-custom-msaf-effb0-aug03)**

---

## What did I do differently?

I deliberately made the training and testing conditions come from different “worlds”:

- **Train:** 11,500 images (**fixed 256 px and 512 px** resolution pools)
- **Test (External / OOD):** 3,500 images (**variable 190 px–800 px** resolutions)

The objective was to evaluate **generalization** under **different source and resolution conditions**, even though images are resized to a fixed model input size.

---

My Custom Model Developed with MSAF (Multi-Scale Attention Fusion):
Unlike standard architectures, my custom MSAF-EffB0 model uses a dynamic attention mechanism operating across three different feature scales. By integrating Squeeze-and-Excitation (SE) blocks with a Softmax-weighted Scale Attention layer, the architecture adaptively prioritizes the most stable and discriminative features.

The most important finding during training was the model’s intrinsic resistance to aggressive data augmentation. The MSAF mechanism effectively filters “noisy” features coming from distorted scales, allowing the model to reach its theoretical performance ceiling without learning incorrect patterns, even under extreme augmentation scenarios.

---

Another issue discussed in the report is that similar images from the same subject can appear in different splits in the original datasets. This subject leakage can inflate metrics, so evaluation was performed using **external-source test datasets**.


## Models

The experiments visible in the outputs and notebooks included in this repository are:

- Standard backbones (timm / transfer learning): ConvNeXt-Tiny, DenseNet121, EfficientNet-B0, InceptionV3, MobileNetV2, ResNet34, ResNet50
- Hybrid (ensemble / feature fusion):
  - DenseNet121 + EfficientNetB0
  - Swin-T + EfficientNetB0
- Custom architecture: **My Model (based on MSAF + EfficientNetB0)**

---

## Results Summary (OOD Test)

Metrics were compiled from `results/metrics_summary.csv`.

| experiment | accuracy | auc | f1 | recall_sensitivity | precision | kappa |
| --- | --- | --- | --- | --- | --- | --- |
| My Model / aug (p=0.3) | 0.908 | 0.988 | 0.901 | 0.822 | 0.998 | 0.817 |
| Hybrid - DenseNet121 + EfficientNetB0 / aug (p=0.3) | 0.861 | 0.967 | 0.841 | 0.726 | 1.000 | 0.723 |
| Hybrid - DenseNet121 + EfficientNetB0 / no-aug | 0.839 | 0.939 | 0.812 | 0.684 | 1.000 | 0.680 |
| My Model / no-aug | 0.805 | 0.936 | 0.764 | 0.618 | 0.999 | 0.613 |
| Hybrid 2- SwinEff / aug (p=0.3) | 0.795 | 0.975 | 0.748 | 0.599 | 0.997 | 0.593 |
| resnet34 / no-aug | 0.794 | 0.954 | 0.747 | 0.596 | 0.999 | 0.591 |
| densenet121 | 0.785 | 0.984 | 0.732 | 0.578 | 1.000 | 0.573 |
| convnext_tiny | 0.775 | 0.960 | 0.716 | 0.557 | 1.000 | 0.553 |
| Hybrid 2- SwinEff / no-aug | 0.745 | 0.956 | 0.665 | 0.498 | 1.000 | 0.494 |
| resnet50 / no-aug | 0.719 | 0.962 | 0.619 | 0.448 | 1.000 | 0.444 |
| inception_v3 / no-aug | 0.710 | 0.901 | 0.602 | 0.430 | 1.000 | 0.426 |
| efficientnet_b0 | 0.693 | 0.903 | 0.568 | 0.397 | 0.997 | 0.392 |
| mobilenetv2_100 / no-aug | 0.639 | 0.889 | 0.450 | 0.290 | 1.000 | 0.286 |


> Best experiment: **My Model / aug (p=0.3)** — Accuracy **0.908**, AUC **0.988**, F1 **0.901**

---

## Quick Start: Inference on a Single Image

### 1) Installation

```bash
pip install -r requirements.txt
```

### 2) Model weights

All trained models and weights have been published on Hugging Face:

**[🤗 BrainMRI OOD Benchmark Model Collection](https://huggingface.co/collections/Fatihaybasn/brainmri-ood-benchmark-10-architectures-13-checkpoints)**

Each model repository contains safe `model.safetensors` weights, architecture configuration, loading code, metrics, and file-verification records. The weights are not stored in GitHub to avoid inflating the repository.

### 3) Run

```bash
python scripts/universal_infer.py --model "PATH/TO/model.pt" --image "PATH/TO/image.jpg" --device cpu
# or
python scripts/universal_infer.py --model "PATH/TO/model.pt" --image "PATH/TO/image.jpg" --device cuda
```

Details: `scripts/universal_infer_README.txt`

---

## Repository Structure

- `notebooks/` → training and experiment notebooks (raw workflow)
- `scripts/` → inference script and usage documentation
- `results/` → metrics summaries and per-model reports (CSV/TXT)
- `reports/` → project report (DOCX) and dataset/weight links
- `src/` → optional scaffold for packaging the code in the future

---

## Datasets

- Main training source: Mendeley (link available in the report)
- External test (OOD): mixture of Kaggle datasets (links in `reports/dataset_links.txt`)

---

## Citation

The repository root contains `CITATION.cff`. GitHub automatically displays it as “Cite this repository.”

---

## License

MIT (see `LICENSE`)
