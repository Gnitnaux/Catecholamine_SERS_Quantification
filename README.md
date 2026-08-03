# Catecholamine SERS Quantification

Machine-learning code for **two-stage identification and quantification of
catecholamines from paper-based surface-enhanced Raman spectroscopy (SERS)**.
The workflow accompanies the manuscript *Two-Stage Machine Learning-Assisted
Multiplex Catecholamine Quantification in Artificial Urine on SERS Paper*.

## Overview

Dopamine (DA), epinephrine (E), and norepinephrine (NE) have closely related
molecular structures and strongly overlapping Raman signatures. This project
combines spectra acquired using an NTA-Fe-functionalized SERS paper with a
two-stage machine-learning workflow:

1. a self-supervised BYOL encoder and a classifier determine which
   catecholamine components are present;
2. single-component samples are quantified from the normalized 1480 cm⁻¹
   marker intensity, while multi-component samples are routed to
   class-specific partial least-squares regression (PLSR) models.

The repository also provides independent PLSR validation and a comparison of
PLSR, random forest (RF), and one-dimensional convolutional neural network
(CNN) regressors. Data splitting is performed at the complete concentration-
group level so that replicate spectra from the same sample group do not leak
between training and validation sets.

**Publication:** the article link will be added after publication.

## Authors

- Xuanting Liu
- ChatGPT CODEX

## Repository structure

```text
.
├── main.py                 # Command-line entry point
├── requirements.txt       # Python dependencies
├── src/                    # Data processing and model implementations
├── data/                   # Input SERS spectra
├── models/                 # Saved model weights and serialized estimators
├── model_output/           # Fallback model-output directory
├── logs/                   # BYOL training logs
├── reports/                # Numerical results exported as CSV files
└── visualizations/         # Confusion matrices and regression plots
```

The analysis commands create model artifacts, logs, numerical reports, and
figures in their corresponding directories. `model_output/` is used as a
fallback when the configured model directory is not writable.

## Source-code guide

| File | Purpose |
| --- | --- |
| `main.py` | Parses command-line arguments and dispatches the four supported analysis modes. |
| `src/utils.py` | Shared constants, deterministic random seeding, folder-name parsing, CSV loading, spectral normalization, group-aware splitting, metrics, result tables, and plotting. |
| `src/ca_paper_plsr.py` | Class-routed quantification: linear 1480/920 cm⁻¹ calibration for single analytes and multi-output PLSR for mixtures. Exports holdout predictions, summaries, plots, and a serialized model. |
| `src/ca_paper_regression.py` | Benchmark implementation comparing PLSR, RF, and 1D-CNN regressors on binary and ternary mixtures. Reports sample- and group-level metrics. |
| `src/byol_model.py` | Spectral augmentation, peak/span masking, CNN-Transformer encoder, BYOL pretraining, PCA-guided and variance regularization, embedding-based classification, and the end-to-end classifier-to-PLSR pipeline. |

The BYOL encoder combines multi-scale convolutions, residual convolutional
blocks, channel attention, positional encoding, and a four-layer Transformer.
Its 256-dimensional embedding is used by a standardized PCA/logistic-
regression classifier. The default random seed is `2026`; the full-pipeline
group split uses seed `3`.

## Dataset organization

Organize the spectra under `data/` using the following layout:

```text
data/
├── data_MPAU_mix/                     # Mixtures in artificial urine
│   ├── 0uM_0uM_0uM_1/                # Blank/artificial-urine background
│   │   ├── spectrum_001.csv
│   │   └── spectrum_002.csv
│   ├── 10uM_0uM_0uM_1/               # 10 µM DA, 0 µM E, 0 µM NE
│   ├── 10uM_5uM_0uM_1/               # 10 µM DA, 5 µM E, 0 µM NE
│   └── 10uM_5uM_15uM_1/              # Ternary mixture
└── data_Water_mix/                    # Same organization for water samples
    └── <DA>uM_<E>uM_<NE>uM_<group>/
        └── spectrum.csv
```

The first three numeric fields in each concentration-folder name always mean
`DA`, `E`, and `NE`, in that order. Decimal concentrations are accepted. The
final field is a group/replicate identifier and is not used as a concentration.
A folder with three zero concentrations is labeled as the background class
`BA`; otherwise the class is generated from the non-zero components (for
example, `DA+NE`).

Each individual spectrum must be a CSV file with at least two columns:

```csv
Raman Shift,Processed Intensity
331.095,21.3305
333.909,89.6749
336.721,80.1411
```

The loader uses the first column as Raman shift and the second column as
intensity, keeps the 330–1600 cm⁻¹ region, removes non-numeric rows, and
interpolates spectra onto the first valid Raman-shift axis when necessary.
Multiple CSV files in one concentration folder are treated as replicate
spectra belonging to the same group and are kept together during data
splitting. Aggregate files placed directly in a dataset root are not consumed
by the training pipeline.

## Usage

### 1. Install dependencies

Python 3.10 or later is recommended. From the repository root, install all
packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

PyTorch automatically uses CUDA when a compatible GPU build is installed and
a CUDA device is available; otherwise the code runs on CPU. BYOL pretraining is
substantially faster on a GPU.

### 2. Run an analysis workflow

Run commands from the repository root and always select an explicit mode.
`--data-dir` is a directory name relative to `data/`, while `--model-dir` may
be a relative or absolute model-output path.

#### 2.1 Class-routed PLSR validation

```bash
python main.py --mode ca_paper_plsr_unmixing \
  --data-dir data_MPAU_mix --model-dir models
```

This mode performs a 70/30 group-aware holdout. It applies a linear
`I1480/I920` calibration to single-component classes and class-specific,
multi-output PLSR to mixtures. Results are written to `reports/`, the
predicted-versus-true plot to `visualizations/`, and the fitted payload to
`models/ca_paper_plsr_unmixing.joblib`.

#### 2.2 Compare PLSR, RF, and CNN quantification

```bash
python main.py --mode ca_paper_unmixing_models \
  --data-dir data_MPAU_mix --model-dir models
```

This benchmark retains binary and ternary mixtures whose non-zero component
concentrations are in the 10–20 µM range. It exports per-method sample/group
predictions, RMSE/R²-related summaries, model tables, radar-chart data, and
prediction plots. The combined model payload is saved as
`models/ca_paper_unmixing_models.joblib`.

#### 2.3 BYOL pretraining and composition classification

Start a fresh BYOL training run:

```bash
python main.py --mode byol_pipeline \
  --data-dir data_MPAU_mix --model-dir models --re_training True
```

Resume an available checkpoint, or train normally when no checkpoint exists:

```bash
python main.py --mode byol_pipeline \
  --data-dir data_MPAU_mix --model-dir models --re_training False
```

Stage 1 generates augmented/masked and clean views of each spectrum and trains
the BYOL encoder. Stage 2 extracts embeddings and performs composition
classification. Checkpoints and encoders are stored in `models/`; epoch logs,
classification tables, and confusion matrices are stored in `logs/`,
`reports/`, and `visualizations/`, respectively.

#### 2.4 Complete two-stage classification and quantification

For the first run on a dataset, train the group-split-specific encoder:

```bash
python main.py --mode CA_Paper_Full_Pipeline \
  --data-dir data_MPAU_mix --model-dir models --re_training True
```

After that encoder exists, reuse it with:

```bash
python main.py --mode CA_Paper_Full_Pipeline \
  --data-dir data_MPAU_mix --model-dir models --re_training False
```

This is the manuscript workflow. It trains the composition classifier using
only training-group embeddings, averages classification probabilities within
each replicate group, routes the group to its predicted composition-specific
calibration model, and then predicts DA/E/NE concentrations. Outputs include:

- `reports/CA_Paper_Full_Pipeline_Classification.csv`
- `reports/CA_Paper_Full_Pipeline_Sample.csv`
- `reports/CA_Paper_Full_Pipeline_Group.csv`
- `reports/CA_Paper_Full_Pipeline_Summary.csv`
- `visualizations/CA_Paper_Full_Pipeline_Confusion.png`
- `visualizations/CA_Paper_Full_Pipeline_Pred_vs_True.png`
- `models/ca_paper_full_pipeline.joblib`

Use `data_Water_mix` in place of `data_MPAU_mix` to run the same supported
workflow on the water dataset.

## Reproducibility notes

- Keep all replicate spectra from a physical/concentration group in the same
  folder; the code relies on this structure to prevent data leakage.
- This software is intended for research use and is not a clinical diagnostic
  device.

## License

This project is released under the [MIT License](LICENSE).
