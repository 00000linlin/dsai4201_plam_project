# Contactless Palmprint ROC/CMC Evaluation (Python)

This folder contains a Python reimplementation of the CRC-CompCode evaluation pipeline for contactless palmprint ROI images.
The default protocol uses `../roi/session1` as the gallery and `../roi/session2` as the probe.

## 1. Folder Structure

Expected workspace layout:

```text
dsai2/
	code/
		run_roc_cmc.py
		requirements.txt
	roi/
		session1/*.bmp
		session2/*.bmp
```

Notes:
- Images should be grayscale `.bmp` ROI files.
- Filenames are expected to be sortable and identity-aligned across sessions.
- With Tongji naming, every 10 images correspond to one palm identity.

## 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:
- `numpy`
- `pillow`
- `matplotlib`

## 3. Run the Evaluation

From this `code/` directory:

```bash
python run_roc_cmc.py
```

The script will:
- Build gallery dictionary features from `session1`
- Score probe samples from `session2` using CRC-RLS residuals
- Compute CMC, ROC, Rank-1/5/10, and EER
- Save plots and numeric outputs

## 4. Optional Arguments

You can override default paths and parameters:

```bash
python run_roc_cmc.py \
	--gallery <path_to_session1> \
	--probe <path_to_session2> \
	--out <output_folder> \
	--patch-size 14 \
	--lambda 1.35 \
	--sigma 4.85 \
	--ratio 1.92 \
	--wavelength 14.1 \
	--samples-per-palm 10
```

Parameter meanings:
- `--patch-size`: patch size for local CompCode histograms
- `--lambda`: regularization in CRC-RLS
- `--sigma`, `--ratio`, `--wavelength`: Gabor filter settings
- `--samples-per-palm`: number of samples per identity in each session

## 5. Output Files

By default, outputs are saved to `results_py/`:

- `roc_curve.png` (main ROC)
- `roc_curve_logx.png` (log-x ROC for low FAR region)
- `roc_curve_zoom.png` (zoomed ROC)
- `cmc_curve.png` (CMC curve)
- `roc_cmc_results.npz` (numeric arrays and summary metrics)

`roc_cmc_results.npz` includes:
- `far`, `gar`, `thresholds`
- `cmc`, `probe_ranks`
- `genuine_scores`, `impostor_scores`
- `eer`, `rank1`, `rank5`, `rank10`

## 6. Reproduction Notes (Appendix)

This experiment is reproduced in Python (Anaconda-compatible), with `numpy`, `pillow`, and `matplotlib` as core dependencies.
Run `run_roc_cmc.py` first; it uses `session1` as gallery and `session2` as probe.
After execution, ROC/CMC figures and result files are generated under `results_py`.
For better inspection in the low-error region, also use `roc_curve_logx.png` and `roc_curve_zoom.png`.

## 7. Troubleshooting

- If you see `No .bmp files found`, check folder paths and extensions.
- If gallery and probe counts differ, ensure both sessions are complete.
- If memory/runtime is high, try running on a machine with more RAM or reduce dataset size for a quick sanity test.