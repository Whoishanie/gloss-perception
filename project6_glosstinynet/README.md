# Project 6 — GlossTinyNet

A lightweight CNN trained from scratch to predict human gloss ratings from surface images.

## Architecture

```
Input (B, 3, 128, 128)
  → Conv1: 3→16 filters, 3×3, ReLU, MaxPool   → (B, 16, 64, 64)
  → Conv2: 16→32 filters, 3×3, ReLU, MaxPool  → (B, 32, 32, 32)
  → Flatten                                    → (B, 32768)
  → FC1: 32768→128, ReLU
  → FC2: 128→1                                 → scalar gloss prediction
```

**Total parameters:** 4,199,649

## Results

| Metric | Value |
|--------|-------|
| Spearman ρ | **0.69** |
| p-value | < 0.001 |
| Training loss (MSE) | decreasing over 10 epochs |

## How to Run

1. Open `GlossTinyNet_Project6.ipynb` in Jupyter or VS Code
2. Update the `BASE` path to point to your data folder
3. Run All cells

## Key Sections

| Section | Description |
|---------|-------------|
| 1 | Imports |
| 2 | Custom Dataset class |
| 3 | Data loading & preprocessing |
| 4 | Model architecture |
| 5 | Training loop |
| 6 | Spearman correlation evaluation |
| 7 | Scatter plot + RDM heatmap |
| 8 | Conv1 filter visualisation |
| 9 | Results & Discussion |
