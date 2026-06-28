# Gloss Perception Modelling with Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Hanieh Amiryousefi  
**Email:** hanieh.amiryousefi.varnosfaderani@stud.uni-giessen.de  
**LinkedIn:** [hanieh-amiryousefi](https://www.linkedin.com/in/hanie-amiryousefi)  
**University:** Justus Liebig University Giessen

---

## Overview

This repository contains two deep learning projects investigating **how well neural networks capture human gloss perception**, the way we judge how shiny a surface looks.

The core question is: *can a neural network learn to rank surfaces by glossiness in the same order as human observers?* This is called **biological plausibility** and is measured using Pearson / Spearman correlation between model predictions and human ratings.

---

## Repository Structure

```
gloss-perception/
├── project6_glosstinynet/
│   ├── GlossTinyNet_Project6.ipynb    ← CNN trained from scratch
│   └── README.md
│
├── extra_assignment_layer_probing/
│   ├── Extra_Assignment_v2.ipynb      ← ResNet-50 linear probing
│   └── README.md
│
├── requirements.txt
└── README.md                          ← you are here
```

---

## Projects

### Project 1 ,GlossTinyNet: Training a CNN from Scratch

> *Can a small CNN trained directly on human gloss ratings learn a biologically plausible representation?*

A lightweight 4-layer CNN trained end-to-end to predict human gloss ratings from 3888 stimulus images.

| Metric | Value |
|--------|-------|
| Architecture | 2 Conv + 2 FC layers |
| Parameters | 4.2 million |
| Loss | MSE |
| Spearman ρ (test) | **0.69** |
| p-value | < 0.001 |

**Key findings:**
- Despite its small size, GlossTinyNet ranked surfaces in broadly the same order as human observers
- The learned Conv1 filters showed orientation-tuned, edge-like structure similar to V1 simple cells
- The RDM revealed structured clustering of stimuli by gloss level

---

### Project 2 , Layer Probing ResNet-50 for Gloss Perception

> *Which layer of a pretrained ResNet-50 best captures human gloss perception — and does it match or exceed a model trained specifically on gloss?*

A linear probing analysis using [Osculari](https://osculari.readthedocs.io/en/latest/) to extract features from 4 ResNet-50 layers and train a Ridge regression probe on each.

| Layer | Depth | Mean Pearson r | 95% CI |
|-------|-------|---------------|--------|
| block0 | Early | 0.4085 | [0.393, 0.425] |
| block1 | Early-Mid | 0.5430 | [0.534, 0.553] |
| **block2** | **Mid** | **0.5824** | **[0.575, 0.590]** |
| block3 | Late | 0.5607 | [0.547, 0.574] |

Results averaged over **10 random seeds** (80/20 train/test split).

**Key findings:**
- Clear **inverted-U pattern** across layers — mid-level features best predict gloss
- block2 achieves the highest Pearson r (0.582) with the smallest std (0.012), making it the most stable and predictive layer
- A pretrained ImageNet model with no gloss-specific training approaches the performance of GlossTinyNet (0.582 vs 0.687), suggesting gloss-relevant features emerge as a by-product of object recognition

---

## Installation

```bash
# Clone the repository
git clone https://github.com/whoishanie/gloss-perception.git
cd gloss-perception

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

The dataset used is the **Tiny Gloss Dataset** (Takuma et al.), consisting of:
- 3888 rendered surface images with varying gloss levels
- Human subjective gloss ratings (`humanlabel.csv`)
- Physical ground-truth gloss values (`groundtruthlabel.csv`)

> **Note:** The dataset is not included in this repository. Please contact the authors or your institution for access.

---

## Requirements

See `requirements.txt`. Main dependencies:

```
torch>=2.0
torchvision>=0.15
osculari
scikit-learn
scipy
pandas
numpy
matplotlib
seaborn
Pillow
```

---

## Results Summary

| Model | Approach | Pearson r | Spearman ρ |
|-------|----------|-----------|------------|
| GlossTinyNet | Trained from scratch on gloss data | — | 0.69 |
| ResNet-50 block2 | Linear probe, pretrained ImageNet | 0.582 | — |

The inverted-U pattern across ResNet-50 layers suggests that **gloss perception is a mid-level visual phenomenon**, consistent with the role of surface texture and specular highlight structure in human gloss judgements (Fleming et al., 2003).

---

## References

- Fleming, R. W., Dror, R. O., & Adelson, E. H. (2003). Real-world illumination and the perception of surface reflectance properties. *Journal of Vision*, 3(5), 347–368.
- Yamins, D. L., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience*, 19(3), 356–365.
- Kriegeskorte, N. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.

---

## License

MIT License — feel free to use and adapt with attribution.
