"""
Statistical analysis for ASD vs HC:
1) Paired t-test on predicted CV and visualization.
2) Column-wise two-sample KS tests between ASD and HC subject-level matrices.

Inputs
------
- cv_glm_results_all.csv : produced by the CV–mean GLM step; must include:
    ['HC_CV (%)','HC_CV_predicted','ASD_CV (%)','ASD_CV_predicted'].
  The file should also have a region identifier (column 'Region') OR an index
  that can be renamed to 'Region'. If neither exists, an ordinal Region_* label
  will be generated.

- asd_mode.csv : subject-level matrix for ASD (columns = networks/features).
- hc_mode.csv  : subject-level matrix for HC  (columns = networks/features).

Notes
-----
- Paired t-test matches regions one-to-one (ASD vs HC predicted CV).
- KS tests compare full column distributions across subjects in ASD vs HC.
"""

import argparse
import warnings
from pathlib import Path

# Optional plotting deps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, ttest_rel


def ensure_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'Region' column exists. Try to recover from index or Unnamed: 0."""
    if 'Region' in df.columns:
        return df
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Region'})
        return df
    # If index looks like region labels, bring it out
    if df.index.name and df.index.name.lower() in ('region', 'roi', 'label', 'network'):
        df = df.reset_index().rename(columns={df.index.name: 'Region'})
        return df
    # Fallback: generate ordinal labels
    warnings.warn("No 'Region' column found; generating ordinal Region_* labels.")
    df = df.copy()
    df.insert(0, 'Region', [f'Region_{i}' for i in range(len(df))])
    return df


def run_paired_ttest_and_plot(cv_glm_path: Path, outdir: Path, save_fig: bool) -> pd.DataFrame:
    """Paired t-test on predicted CV and optional visualization."""
    df = pd.read_csv(cv_glm_path)
    df = ensure_region_column(df)

    # Sanity check for required columns
    needed = {"ASD_CV_predicted", "HC_CV_predicted"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in {cv_glm_path}: {missing}")

    # Paired t-test (paired by region)
    t_stat, p_value = ttest_rel(df["ASD_CV_predicted"], df["HC_CV_predicted"])
    n_regions = len(df)
    print(f"[Paired t-test] ASD vs HC predicted CV: t = {t_stat:.3f}, p = {p_value:.3g} (n={n_regions})")

    # Save summary
    summary = pd.DataFrame([{
        "n_regions": n_regions,
        "t_stat": t_stat,
        "p_value": p_value
    }])
    summary_path = outdir / "paired_ttest_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot & save
    if save_fig:
        melted = df.melt(
            id_vars="Region",
            value_vars=["ASD_CV_predicted", "HC_CV_predicted"],
            var_name="Group",
            value_name="CV"
        )
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Group", y="CV", data=melted, showfliers=False, palette=["#66c2a5", "#fc8d62"])
        sns.swarmplot(x="Group", y="CV", data=melted, color=".25", size=5)
        plt.title("Paired t-test of Predicted CV between ASD and HC")
        plt.ylabel("Coefficient of Variation (%)")
        plt.xlabel("")
        plt.tight_layout()
        fig_path = outdir / "cv_predicted_boxplot.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[Figure] Saved: {fig_path}")

    return summary


def run_ks_tests(asd_path: Path, hc_path: Path, outdir: Path) -> pd.DataFrame:
    """Column-wise two-sample KS tests between ASD and HC CSVs on common numeric columns."""
    asd_df = pd.read_csv(asd_path)
    hc_df  = pd.read_csv(hc_path)

    # Intersect columns
    common_cols = list(set(asd_df.columns).intersection(set(hc_df.columns)))
    if not common_cols:
        raise ValueError("No common columns between ASD and HC CSVs. Check file schema.")

    rows = []
    for col in sorted(common_cols):
        # Skip non-numeric
        if not (np.issubdtype(asd_df[col].dtype, np.number) and np.issubdtype(hc_df[col].dtype, np.number)):
            continue
        asd_vals = asd_df[col].dropna().values
        hc_vals  = hc_df[col].dropna().values
        if len(asd_vals) == 0 or len(hc_vals) == 0:
            continue
        ks_stat, p_value = ks_2samp(asd_vals, hc_vals)
        rows.append({
            "column": col,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "significant_p_lt_0_05": p_value < 0.05
        })

    results = pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)
    out_csv = outdir / "ks_test_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"[KS] Tested {len(results)} numeric columns. Saved: {out_csv}")
    return results


# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="ASD vs HC deviation statistics (paired t-test + KS tests).")
    parser.add_argument("--cv-glm", type=str, required=True, help="Path to cv_glm_results_all.csv")
    parser.add_argument("--asd", type=str, required=True, help="Path to asd_mode.csv")
    parser.add_argument("--hc",  type=str, required=True, help="Path to hc_mode.csv")
    parser.add_argument("--outdir", type=str, default="output_stats", help="Directory to save outputs")
    parser.add_argument("--save-fig", action="store_true", help="Save boxplot figure for predicted CV")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Paired t-test + (optional) figure
    run_paired_ttest_and_plot(Path(args.cv_glm), outdir, save_fig=args.save_fig)

    # 2) KS tests across common numeric columns
    run_ks_tests(Path(args.asd), Path(args.hc), outdir)


if __name__ == "__main__":
    main()
