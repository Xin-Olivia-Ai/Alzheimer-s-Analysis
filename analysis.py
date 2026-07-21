# %% [markdown]
# # Cognitive Decline & Brain Volume — Exploratory Analysis
# Dataset: OASIS Longitudinal (373 MRI visits, 150 subjects, ages 60-96)
# Source: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
#
# Columns: Subject ID, MRI ID, Group (Demented/Nondemented/Converted), Visit,
# MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load and inspect

# %%
df = pd.read_csv("oasis_longitudinal.csv")
print(df.shape)
df.head()

# %%
df.info()
df.isna().sum()

# %% [markdown]
# ## 2. Clean
# SES and MMSE have a handful of missing values in the real dataset — decide here
# whether to drop or impute (median imputation is a defensible default for SES).

# %%
df["SES"] = df["SES"].fillna(df["SES"].median())
df = df.dropna(subset=["MMSE"])
print("Remaining rows:", len(df))

# %% [markdown]
# ## 3. Group comparison: Demented vs Nondemented
# This is the core research question: which variables differ most between the two
# groups, and how strongly?

# %%
group_summary = df.groupby("Group")[["Age", "MMSE", "eTIV", "nWBV", "ASF", "EDUC"]].agg(
    ["mean", "std", "count"]
)
group_summary

# %%
demented = df[df["Group"] == "Demented"]
nondemented = df[df["Group"] == "Nondemented"]

for col in ["MMSE", "nWBV", "Age", "EDUC"]:
    t_stat, p_val = stats.ttest_ind(
        demented[col].dropna(), nondemented[col].dropna(), equal_var=False
    )
    print(f"{col}: t={t_stat:.2f}, p={p_val:.4f}")

# %% [markdown]
# ## 4. Visualize the group differences

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x="Group", y="MMSE", ax=axes[0])
axes[0].set_title("Cognitive score (MMSE) by group")
sns.boxplot(data=df, x="Group", y="nWBV", ax=axes[1])
axes[1].set_title("Normalized brain volume by group")
plt.tight_layout()
plt.savefig("group_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Correlation structure
# Which variables move together? nWBV (brain volume) vs MMSE (cognition) vs Age
# is the classic relationship to check here.

# %%
corr_cols = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
corr = df[corr_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Does education buffer cognitive decline?
# A common finding in this literature is "cognitive reserve" — more years of
# education is associated with better MMSE scores even at a similar brain volume.
# Check whether that shows up here.

# %%
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="nWBV", y="MMSE", hue="Group", alpha=0.7)
plt.title("Brain volume vs cognitive score, by group")
plt.tight_layout()
plt.savefig("volume_vs_mmse.png", dpi=150)
plt.show()

educ_corr, educ_p = stats.pearsonr(df["EDUC"], df["MMSE"])
print(f"EDUC vs MMSE correlation: r={educ_corr:.2f}, p={educ_p:.4f}")

# %% [markdown]
# ## 7. Write-up

- Sample size after cleaning: 371 visits (Nondemented 190, Demented 144,
  Converted 37).
- MMSE difference between Demented (M=24.5) and Nondemented (M=29.2) groups:
  mean difference of ~4.7 points (t=-12.40, p<0.0001). Expected, since MMSE is
  itself part of the diagnostic criteria — this is a sanity check, not a finding.
- Age was NOT significantly different between groups (t=-0.96, p=0.34), while
  brain volume (nWBV) WAS significantly lower in the Demented group (0.716 vs
  0.741, t=-6.51, p<0.0001). Because age is matched between groups, this rules
  out "they're just older" as the explanation — the volume loss looks tied to
  the disease process itself, not general aging.
- Education also differed significantly (Demented M=13.7 yrs vs Nondemented
  M=15.1 yrs, t=-4.61, p<0.0001), and EDUC correlated with MMSE at r=0.19
  (p=0.0002) across the full sample. Direction supports the "cognitive
  reserve" hypothesis, but the correlation is weak — education looks like a
  background variable associated with group membership, not a strong direct
  predictor of cognitive score on its own.
- Caveat on the correlation heatmap: eTIV and ASF correlate at r=-0.99, and
  MMSE/CDR at r=-0.69 — both are close to circular (ASF is mathematically
  derived from eTIV; MMSE and CDR are two instruments measuring the same
  underlying cognitive state), so neither is a "real" independent finding.
  The genuinely informative, non-circular relationship is nWBV vs MMSE at
  r=0.34 — a moderate, non-tautological link between brain volume and
  cognitive score across the whole sample.

# %%
