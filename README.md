# Does Brain Volume Loss Track with Cognitive Decline, Independent of Age?

An exploratory data analysis of the OASIS longitudinal MRI/clinical dataset (373 visits,
150 subjects, ages 60–96), testing whether brain-volume loss in dementia patients is a
real, independent signal — or just a proxy for "these subjects are older."

## Key findings

- **Age was not significantly different** between Demented and Nondemented groups
  (t=-0.96, p=0.34) — ruling out age as a confound.
- **Brain volume (nWBV) was significantly lower** in the Demented group (0.716 vs. 0.741,
  t=-6.51, p<0.0001), with age matched between groups. This points to volume loss tied to
  the disease process itself, not general aging.
- **Education correlated weakly with cognitive score** (r=0.19, p=0.0002) — direction
  consistent with the "cognitive reserve" hypothesis, but not a strong predictor on its
  own.
- Two of the most eye-catching correlations in the raw data (eTIV to ASF at r=-0.99,
  MMSE to CDR at r=-0.69) turned out to be near-circular — each pair measures the same
  underlying quantity two different ways, not an independent relationship. The one
  genuinely independent signal was **nWBV vs. MMSE at r=0.34**.

Full write-up with methodology is in [`analysis.py`](./analysis.py).

## Dataset

[OASIS Longitudinal MRI Data](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)
(150 subjects, 373 imaging visits). Not included in this repo — see below to reproduce.

## Reproduce this analysis

```bash
pip install pandas matplotlib seaborn scipy
```

1. Download `oasis_longitudinal.csv` from the Kaggle link above and place it in this
   folder.
2. Open `analysis.py` in VS Code or JupyterLab (it uses `# %%` cell markers, so both
   tools will run it cell-by-cell) and run top to bottom.

## Tech stack

Python, pandas, seaborn/matplotlib, scipy (Welch's t-tests, Pearson correlation)

## Files

- `analysis.py` - full analysis: cleaning, group comparisons, correlation analysis,
  write-up
- `correlation_heatmap.png`, `Figure_1.png`, `volume_vs_mmse.png` - output visualizations
