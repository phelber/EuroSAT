# EuroSAT Land-Cover Classification

Land-cover classification of 64 × 64 Sentinel-2 image patches into 10 classes, for the
Machine Learning coding challenge at the University of St. Gallen (Kaggle competition
`7-854-1-00-machine-learning-2026-coding-challenge`).

**The core difficulty:** the training data (EuroSAT) is Sentinel-2 **Level-1C**, with haze,
while the Kaggle test data is **Level-2A**, atmospherically corrected. Models reach about 99% on
EuroSAT but far less on Kaggle. Most of the work in this repository is about closing that gap.

## Quick start for the team

1. Set up the environment, the Kaggle token and the data: see [Setup](#setup) below (once, about 15 minutes).
2. Open `notebooks/00_results_overview.ipynb` for where the project stands.
3. Read the notebooks in order: `data/01` → `09`, then `experiments/E01` → `E12`. Each one starts
   with a short explanation and ends with its result.

### Where do I…?

| I want to… | Go to |
|---|---|
| see all results and what we learned | `notebooks/00_results_overview.ipynb` |
| understand the data and the train/test difference | `notebooks/data/01_eda.ipynb` |
| see how the haze correction works | `notebooks/data/04_haze_correction.ipynb`, code in `src/preprocessing.py` |
| reproduce a Kaggle score exactly | the experiment's notebook in `notebooks/experiments/`, run as is (`LOAD_RUN` / `LOAD_RUNS` load the saved models) |
| try a new idea | copy the closest `E…` notebook as the next number, change its settings block, set `LOAD_RUN = None` |
| change the model, training loop or augmentation | `src/models.py`, `src/training.py` |
| add a new data correction | `src/preprocessing.py` or `src/fourier.py`, then fit and save it in a `data/` notebook |
| submit to Kaggle | [Submitting to Kaggle](#submitting-to-kaggle) below |

## Results so far

| Notebook | Change | Internal test | Kaggle public |
|---|---|---|---|
| E01 | EfficientNet-B0, 12 bands, no correction | 98.79% | 56.48% |
| E02 | + histogram matching of training images (`hist_split`) | 98.64% | 64.06% |
| E03 | E02 + AdaBN on Kaggle images, water share matched (no retraining) | – | 65.28% |
| E04 | E02 + contrast/brightness augmentation | 98.77% | 64.39% |
| E04 | E02 + contrast/brightness augmentation + AdaBN balanced | – | 68.18% |
| E05a | E04 + Fourier A: FDA augmentation (+ AdaBN) | 98.8% | 66.26% |
| E05b | E04 + Fourier B: spectral matching (+ AdaBN) | 98.6% | 67.85% |
| E05c | E04 + Fourier C: low-frequency equaliser layer (+ AdaBN) | 98.8% | 67.90% |
| E06a | E04 + pseudo-labelled Kaggle images, round 1 (150 per class) + AdaBN balanced | 98.96% | 70.71% |
| E06b | round 2: pseudo-labels from E06a (up to 250 per class); without AdaBN | 98.81% | 70.99% |
| E06c | E06b with pseudo-labels used 2× per epoch + AdaBN balanced (without AdaBN 71.08%) | 98.94% | 71.50% |
| E07a | E06c with shape-focused input (per-image standardisation + statistics branch) + AdaBN balanced | 98.17% | 70.24% |
| E07a | ensemble: E06c + E07a (averaged probabilities) | – | 70.99% |
| E07b | E07a + topological features (persistent homology, `data/07`) + AdaBN balanced | 97.53% | – |
| E08a | model soup: weights of E06b + E06c averaged (no training) + AdaBN balanced | 99.0% | 71.69% |
| E08b | E06c setup with EMA weight averaging (+ checkpoints for "training less") + AdaBN balanced | 98.8% | 71.55% |
| E09a | round 3: all pseudo-labels whose 8 views agree, single teacher E06c + AdaBN, **no AdaBN** | 98.86% | **72.67%** |
| E09a | same model + AdaBN balanced | – | 71.78% |
| E09b | E09a + soft pseudo-labels (teacher probabilities) for the uncertain images | 98.62% | – |
| E10a | **small track:** MobileNetV3-Small (934k parameters), E09a setup, no AdaBN (with AdaBN 70.61%) | 98.02% | 70.94% |
| E11a | two-branch network: EfficientNet-B0 + MobileNetV3-Small behind one input layer (4.96M), no AdaBN (with 71.27%) | 98.72% | 71.92% |
| E12a | round 4: pseudo-labels from E09a (no AdaBN), no AdaBN (with AdaBN 72.39%) | 98.32% | 72.34% |

Medium track (limit 10M): EfficientNet-B0 4,022,950 parameters (E07a: 4,029,350; E07b: 4,058,854; E11a: 4,957,014).
Small track (limit 1M): MobileNetV3-Small 934,074 parameters (E10a). Since E09a, AdaBN no longer helps (see `00_results_overview`, lesson 12). `notebooks/00_results_overview.ipynb`
builds this table and charts automatically from `runs/`.

## Repository structure

```
notebooks/
  00_results_overview.ipynb   all runs and scores, charts, lessons learned
  data/          data analysis and preprocessing (01–09)
  experiments/   one notebook per experiment (E01 … E12; E05–E09 with variants a, b, c), each linked to its run folders
src/           shared Python code imported by the notebooks
scripts/       data download scripts
artifacts/     small results of the data notebooks, shared via Git
  splits/        train/val/test split (03_split)
  preprocessing/ fitted haze and Fourier corrections and their comparisons (04, 05)
  quality_flags.csv  data quality flags (02_data_quality)
  kaggle_confidence.csv  per-image confidence of the best models on Kaggle (06), used for pseudo-labels
  kaggle_water_vegetation.csv  water and vegetation measures per Kaggle image (08)
  probes/        leaderboard probe submissions of 09_label_convention
runs/          one folder per training run, shared via Git (weights only for submitted runs)
data/          raw data and caches, NOT in Git (download with scripts/)
```

**Rule of thumb:** `data/` holds what can be downloaded again. `artifacts/` and `runs/` hold what
we created ourselves and need to share.

### Notebooks

| Notebook | Purpose |
|---|---|
| `00_results_overview` | All runs, Kaggle scores, domain-gap chart, lessons learned (reads `runs/`) |
| **`data/`** | |
| `01_eda` | Exploratory analysis: class balance, band order, brightness and resolution differences between train and test |
| `02_data_quality` | Quality checks (duplicates, broken bands, outliers, haze). Nothing removed, unusual images flagged |
| `03_split` | Stratified 70/15/15 train/val/test split; neighbouring patches kept in the same split |
| `04_haze_correction` | Compares haze corrections (linear, histogram matching variants). No training needed |
| `05_fourier_analysis` | Frequency analysis of the remaining train/test difference; compares three Fourier methods without training |
| `06_kaggle_confidence` | Which Kaggle images the best model is unsure about (classes, pairs, examples); basis for pseudo-labelling |
| `07_topological_features` | Persistent homology (`gudhi`): do shapes transfer better than band values? Topological features vs spectral statistics |
| `08_vegetation_shift` | Water and vegetation on Kaggle vs EuroSAT: uncertain SeaLake images, greener and wetter crops, which class cues survive |
| `09_label_convention` | Does Kaggle label mixed water/vegetation scenes differently? EuroSAT's rule per water share, three leaderboard probes |
| **`experiments/`** | |
| `E01_baseline` | Baseline model, reference for all improvements |
| `E02_hist_split` | Baseline with the best haze correction (`hist_split`) |
| `E03_adabn` | AdaBN: re-measures the BatchNorm statistics on Kaggle images (no training) |
| `E04_contrast_aug` | Contrast/brightness augmentation during training, with and without AdaBN |
| `E05_fourier` | Fourier methods: a FDA augmentation, b spectral matching, c low-frequency equaliser layer (each + AdaBN) |
| `E06_pseudo_labelling` | Training with pseudo-labelled Kaggle images: a round 1, b round 2, c round 2 with 2× weight |
| `E07_shape_input` | Shape-focused input (per-image standardisation): a without, b with topological features; ensembles with E06c |
| `E08_weight_averaging` | a model soup (averaged weights of E06 runs, no training); b one run with checkpoints, EMA and SWA |
| `E09_pseudo_labels_r3` | Round 3 from one teacher: a all images whose 8 views agree (best model), b plus soft labels |
| `E10_small_track` | Small track (≤ 1M parameters): MobileNetV3-Small with the E09a setup |
| `E11_two_branch` | One network with an EfficientNet-B0 and a MobileNetV3-Small branch behind one input layer, started from E09a + E10a |
| `E12_pseudo_labels_r4` | Round 4 with E09a (no AdaBN) as teacher |

Each experiment notebook starts with its settings (`CORRECTIONS`, `cfg`, ...), so what it changes is
visible at the top; the shared steps live in `src/experiment.py` (`finish_run` does internal test, Kaggle
predictions, saving and AdaBN in one call). E05–E09 hold several variants of one idea, each in its own section, with a
comparison at the end; `LOAD_RUNS` maps every variant to its run folder. A new experiment: copy the closest `E…`
notebook, give it the next number (or add a variant), change the settings, set its `LOAD_RUN(S)` entry to `None`.

Experiment notebooks have a `LOAD_RUN` setting: with a run folder set, they skip training, load the
saved weights (`model.pt` from Git) and recompute all results and the exact submission file in
about a minute. Set `LOAD_RUN = None` to train a new model; retraining gives a very similar but not
identical model (GPU calculations are not bit-for-bit reproducible), so Kaggle scores can differ
by about 1–2 points.

### `src/`

| File | Content |
|---|---|
| `data.py` | Band orders and mapping, class list, paths, loading (with a fast `.npy` cache in `data/cache/`) |
| `preprocessing.py` | Haze corrections: histogram matching (`HistogramMatcher`, `SplitMatcher`), linear percentile matching |
| `models.py` | Model builders; adapts pretrained RGB weights to 12 bands; MobileNetV3-Small (small track); shape-focused model (`ShapeFocusedNet`); two-branch network (`TwoBranchNet`); track parameter limits |
| `experiment.py` | Shared steps of the experiment notebooks: loading and correcting data, AdaBN, `finish_run`, pseudo-label selection, variant comparison, plots |
| `training.py` | GPU training loop (hard or soft/weighted targets), early stopping, optional EMA and checkpoints, prediction with test-time augmentation, saving and loading runs |
| `fourier.py` | Fourier methods: FDA augmentation, spectral matching, low-frequency equaliser layer |
| `topology.py` | Topological features (persistent homology with `gudhi`) for `07_topological_features` and `E07b` |
| `utils.py` | Device selection (CUDA → Apple GPU → CPU), seeding, parameter count |
| `plot_style.py` | Shared plot colours and style |

### A run folder

```
runs/<date>_<name>/
  config.json     model, bands, preprocessing, normalisation, hyperparameters
  history.csv     loss, accuracy and Kaggle monitoring per epoch
  metrics.json    validation / internal test accuracy, Kaggle public score
  submission.csv  the file submitted to Kaggle
  model.pt        trained weights (EfficientNet-B0 16 MB, MobileNetV3-Small 4 MB, two-branch 19 MB; in Git only for runs submitted to Kaggle)
  variants/       E08b only: checkpoints, EMA and SWA weights (not in Git) with their own submission/metrics
```

## Setup

macOS, from the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name eurosat --display-name "EuroSAT (.venv 3.14)"
```

In VS Code, select the kernel **EuroSAT (.venv 3.14)** for the notebooks.

### Kaggle token

1. Join the competition on Kaggle and accept its rules.
2. Kaggle → Settings → API → **Generate New Token**. Copy the token value (starts with `KGAT_`;
   it is shown only once). The token *name* is not the token.
3. Save it, without echoing it to the terminal:

```bash
mkdir -p ~/.kaggle
read -s "TOKEN?Paste Kaggle token: " && printf '%s' "$TOKEN" > ~/.kaggle/access_token && unset TOKEN
chmod 600 ~/.kaggle/access_token
```

Never commit tokens. (`kagglehub.login()` only keeps the token for the current Python process,
so it does not work for the download scripts.)

### Download data

```bash
python scripts/download_data.py --ms-only      # EuroSAT 13-band GeoTIFFs (~2 GB)
python scripts/download_data.py --kaggle-only  # Kaggle test set and CSVs
```

| Command | Description |
| --- | --- |
| `python scripts/download_data.py` | Everything (also the RGB version, which the notebooks don't need) |
| `python scripts/download_data.py --eurosat-only` | Both EuroSAT datasets |
| `python scripts/download_data.py --rgb-only` | Only the RGB EuroSAT dataset |
| `python scripts/download_data.py --force` | Force re-download |

The first notebook run builds `data/cache/` (about 3 GB, one minute).

## Hardware

Training uses the GPU automatically: CUDA if available, otherwise the Apple GPU (MPS), otherwise
the CPU. On an M4 MacBook one training run (20 epochs) takes about 18 minutes; on the CPU it would
take days. The MacBook Air throttles under long load, so expect roughly 20–40% longer.

Distributing one run over several laptops is not worth it. Run **different experiments in
parallel**, one per teammate.

## Submitting to Kaggle

```bash
kaggle competitions submit -c 7-854-1-00-machine-learning-2026-coding-challenge \
  -f runs/<run>/submission.csv -m "<what changed>"
kaggle competitions submissions -c 7-854-1-00-machine-learning-2026-coding-challenge
```

Limit: 15 submissions per day. Write the public score into the run's `metrics.json`.

## Team workflow

- **One change per experiment**, each in its own `E…` notebook and run folder, so every
  Kaggle score can be traced to exactly one change. Re-run `00_results_overview` afterwards.
- **Don't edit the same notebook at the same time**: notebooks are hard to merge in Git.
- Notebooks are committed **with outputs** (they are the documentation). Run them top to bottom
  before committing.
- **Weights of every run submitted to Kaggle go into Git.** `model.pt` files are ignored by default;
  add a submitted run's weights explicitly with `git add -f runs/<run>/model.pt`. Then anyone can
  reproduce the exact model and score with `LOAD_RUN`. Also add the weights of a teacher model that a notebook
  loads (e.g. the E06c + AdaBN teacher of E09/E12). Weights of other runs stay local. Exception: the submitted
  E06a/E06b/E06c (no AdaBN), E07a and E08a weights were not committed, so those notebooks can only retrain.

## Data sources

- EuroSAT, official Zenodo record: https://zenodo.org/records/7711810
  (`EuroSAT_MS.zip`: 13 Sentinel-2 bands, 27,000 labelled images, 10 classes). MIT license,
  based on Copernicus Sentinel data.
- Kaggle competition `7-854-1-00-machine-learning-2026-coding-challenge`: Level-2A test set
  (4,232 images, 12 bands without B10) and submission format.

## References

1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). "EuroSAT: A Novel Dataset and Deep
   Learning Benchmark for Land Use and Land Cover Classification." IEEE JSTARS.
2. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2018). "Introducing EuroSAT: A Novel Dataset
   and Deep Learning Benchmark for Land Use and Land Cover Classification." IGARSS 2018.
3. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks." ICML. ImageNet weights from torchvision.
4. Last year's solution (team M.A.R.S, 2025) was used as a reference for the problem analysis.
   All code in this repository was written independently.
