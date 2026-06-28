# Extra Assignment — Layer Probing ResNet-50 for Gloss Perception

Linear probing analysis using pretrained ResNet-50 to identify which visual layer best captures human gloss perception.

## Approach

1. Extract features from 4 ResNet-50 layers using [Osculari](https://osculari.readthedocs.io/en/latest/)
2. Train a Ridge regression linear probe on each layer's features
3. Evaluate Pearson r on held-out test set
4. Repeat over 10 random seeds for robustness

## Results

| Layer | Depth | Mean Pearson r | Std | 95% CI |
|-------|-------|---------------|-----|--------|
| block0 | Early | 0.4085 | 0.0258 | [0.393, 0.425] |
| block1 | Early-Mid | 0.5430 | 0.0154 | [0.534, 0.553] |
| **block2** | **Mid** | **0.5824** | **0.0116** | **[0.575, 0.590]** |
| block3 | Late | 0.5607 | 0.0220 | [0.547, 0.574] |

**Best layer: block2** — mid-level features best explain human gloss perception.

## How to Run

1. Open `Extra_Assignment_v2.ipynb` in Jupyter or VS Code
2. Update the `BASE` path to point to your data folder
3. Run All cells (note: ResNet-50 weights ~100MB will download on first run)

## Dependencies

```bash
pip install osculari scikit-learn torch torchvision scipy pandas numpy matplotlib seaborn
```

## Key Sections

| Section | Description |
|---------|-------------|
| 1 | Install & Import |
| 2 | Dataset |
| 3 | Feature extraction with Osculari |
| 4 | Multi-run linear probing (10 seeds) |
| 5 | Results table |
| 6 | Visualisation (bar chart + box plot) |
| 7 | Results & Discussion |
