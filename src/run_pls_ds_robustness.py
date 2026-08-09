#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PLS analysis and differential-stability robustness analysis
============================================================

This script performs transcriptomic partial least squares (PLS) analysis
using Allen Human Brain Atlas (AHBA) gene-expression data processed with
abagen.

Main steps
----------
1. Map AHBA samples to a user-defined atlas using abagen.
2. Retain donor-specific regional gene-expression matrices.
3. Calculate differential stability (DS) for each gene across donors.
4. Generate group-level expression matrices using multiple DS thresholds.
5. Refit PLS independently for DS >= 0, DS >= 0.1, and DS >= 0.2.
6. Compare regional PLS1 score maps across DS thresholds.
7. Save gene-level DS values, PLS1 scores, gene weights, summary
   statistics, and robustness figures.

The primary analysis uses DS >= 0.1.
DS >= 0 and DS >= 0.2 are used as sensitivity analyses.

Dependencies
------------
abagen
numpy
pandas
scipy
scikit-learn
matplotlib
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abagen

from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression


# ============================================================
# Utilities
# ============================================================

def matlab_zscore(x, axis=0):
    """
    Z-score using sample standard deviation (ddof=1),
    consistent with MATLAB zscore().
    """
    x = np.asarray(x, dtype=float)

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, ddof=1, keepdims=True)

    return (x - mean) / std


def prepare_donor_expression(expression_donors):
    """
    Convert abagen donor-specific output to a list of DataFrames.

    abagen output may be returned as either a dictionary or a list,
    depending on the software version.
    """
    if isinstance(expression_donors, dict):
        donor_ids = list(expression_donors.keys())
        donor_list = list(expression_donors.values())

    elif isinstance(expression_donors, (list, tuple)):
        donor_list = list(expression_donors)
        donor_ids = [
            f"donor_{i + 1}"
            for i in range(len(donor_list))
        ]

    else:
        raise TypeError(
            "Unexpected donor-expression format: "
            f"{type(expression_donors)}"
        )

    reference_index = donor_list[0].index
    reference_genes = donor_list[0].columns

    for i, df in enumerate(donor_list):

        if not reference_genes.equals(df.columns):
            raise ValueError(
                "Gene columns are inconsistent across donors."
            )

        if not reference_index.equals(df.index):
            donor_list[i] = df.reindex(reference_index)

    return donor_ids, donor_list


def calculate_differential_stability(donor_list):
    """
    Calculate gene-wise differential stability across AHBA donors.

    Spearman correlations of regional expression patterns are computed
    across donor pairs and averaged for each gene.
    """
    _, stability = abagen.keep_stable_genes(
        donor_list,
        threshold=0,
        percentile=False,
        rank=True,
        return_stability=True
    )

    genes = donor_list[0].columns.astype(str)

    ds = pd.DataFrame({
        "gene": genes,
        "DS": stability
    })

    ds = (
        ds
        .sort_values("DS", ascending=False)
        .reset_index(drop=True)
    )

    return ds


def average_donors(donor_list):
    """
    Average regional gene expression across donors.

    Returns
    -------
    DataFrame
        Regions x genes.
    """
    arrays = [
        df.to_numpy(dtype=float)
        for df in donor_list
    ]

    donor_stack = np.stack(arrays, axis=0)

    group_expression = np.nanmean(
        donor_stack,
        axis=0
    )

    return pd.DataFrame(
        group_expression,
        index=donor_list[0].index,
        columns=donor_list[0].columns.astype(str)
    )


# ============================================================
# PLS
# ============================================================

def run_pls1(
    expression,
    phenotype,
    ds_table,
    ds_threshold
):
    """
    Run one-component PLS using genes satisfying a specified
    differential-stability threshold.

    Parameters
    ----------
    expression : DataFrame
        Region x gene group-level expression matrix.

    phenotype : ndarray
        Regional phenotype vector.

    ds_table : DataFrame
        Gene-wise differential stability.

    ds_threshold : float
        Minimum DS value for gene inclusion.

    Returns
    -------
    dict
        PLS1 scores, gene weights, model statistics, and gene list.
    """

    selected_genes = ds_table.loc[
        ds_table["DS"] >= ds_threshold,
        "gene"
    ].tolist()

    selected_genes = [
        gene
        for gene in selected_genes
        if gene in expression.columns
    ]

    X_df = expression.loc[:, selected_genes]

    X = X_df.to_numpy(dtype=float)
    genes = np.asarray(X_df.columns)

    # --------------------------------------------------------
    # Remove genes containing non-finite values
    # --------------------------------------------------------

    valid_gene = np.all(np.isfinite(X), axis=0)

    X = X[:, valid_gene]
    genes = genes[valid_gene]

    # --------------------------------------------------------
    # Remove genes with zero variance
    # --------------------------------------------------------

    gene_std = np.std(
        X,
        axis=0,
        ddof=1
    )

    valid_gene = (
        np.isfinite(gene_std)
        & (gene_std > 0)
    )

    X = X[:, valid_gene]
    genes = genes[valid_gene]

    # --------------------------------------------------------
    # Normalize predictors and response
    # --------------------------------------------------------

    X = matlab_zscore(X, axis=0)

    Y = matlab_zscore(
        np.asarray(phenotype).reshape(-1, 1),
        axis=0
    ).ravel()

    # --------------------------------------------------------
    # PLS1
    # --------------------------------------------------------

    pls = PLSRegression(
        n_components=1,
        scale=False,
        max_iter=5000,
        tol=1e-10
    )

    pls.fit(
        X,
        Y.reshape(-1, 1)
    )

    region_scores = pls.x_scores_[:, 0].copy()
    gene_weights = pls.x_weights_[:, 0].copy()

    # --------------------------------------------------------
    # Resolve arbitrary component sign
    #
    # Orient PLS1 so that regional PLS1 scores are positively
    # correlated with the phenotype.
    # --------------------------------------------------------

    r_y = pearsonr(
        region_scores,
        Y
    )[0]

    if r_y < 0:
        region_scores *= -1
        gene_weights *= -1
        r_y *= -1

    # --------------------------------------------------------
    # Variance in Y explained by the fitted PLS1 model
    # --------------------------------------------------------

    y_pred = pls.predict(X).ravel()

    ss_res = np.sum(
        (Y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (Y - np.mean(Y)) ** 2
    )

    explained_y = 100 * (
        1 - ss_res / ss_tot
    )

    weight_table = pd.DataFrame({
        "gene": genes,
        "PLS1_weight": gene_weights
    })

    weight_table = weight_table.sort_values(
        "PLS1_weight",
        ascending=False
    )

    return {
        "threshold": ds_threshold,
        "n_genes": len(genes),
        "genes": genes,
        "score": region_scores,
        "weights": weight_table,
        "corr_with_Y": r_y,
        "variance_explained_Y": explained_y
    }


def align_component_sign(reference, target):
    """
    Align the sign of a PLS component to a reference component.
    """
    r = pearsonr(reference, target)[0]

    if r < 0:
        target = -target

    return target


# ============================================================
# Spin test
# ============================================================

def spin_test(map1, map2, permutations):
    """
    Calculate a two-sided spin-test P value.

    Parameters
    ----------
    map1, map2 : ndarray
        Regional brain maps.

    permutations : ndarray
        Permutation indices with shape:
        n_permutations x n_regions.

    Returns
    -------
    observed_r : float
    p_spin : float
    """
    observed_r = pearsonr(map1, map2)[0]

    null_r = np.zeros(
        permutations.shape[0],
        dtype=float
    )

    for i, perm in enumerate(permutations):

        null_r[i] = pearsonr(
            map1,
            map2[perm]
        )[0]

    p_spin = (
        1
        + np.sum(
            np.abs(null_r)
            >= np.abs(observed_r)
        )
    ) / (
        len(null_r) + 1
    )

    return observed_r, p_spin


# ============================================================
# Plot
# ============================================================

def plot_robustness(
    main_score,
    alternative_scores,
    output_file,
    spin_results=None
):
    """
    Plot PLS1 robustness across differential-stability thresholds.
    """

    score_ds0 = alternative_scores[0.0]
    score_ds02 = alternative_scores[0.2]

    r0 = pearsonr(
        main_score,
        score_ds0
    )[0]

    r02 = pearsonr(
        main_score,
        score_ds02
    )[0]

    all_values = np.concatenate([
        main_score,
        score_ds0,
        score_ds02
    ])

    lower = np.min(all_values)
    upper = np.max(all_values)

    padding = (
        upper - lower
    ) * 0.08

    lower -= padding
    upper += padding

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.5, 4)
    )

    comparisons = [
        (
            axes[0],
            score_ds0,
            0.0,
            r0,
            "a"
        ),
        (
            axes[1],
            score_ds02,
            0.2,
            r02,
            "b"
        )
    ]

    for ax, y, threshold, r_value, panel in comparisons:

        ax.scatter(
            main_score,
            y,
            s=32,
            alpha=0.55
        )

        slope, intercept = np.polyfit(
            main_score,
            y,
            1
        )

        xx = np.linspace(
            lower,
            upper,
            200
        )

        ax.plot(
            xx,
            slope * xx + intercept,
            linewidth=1.8
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.set_xlabel(
            "PLS1 score (DS ≥ 0.1)"
        )

        ax.set_ylabel(
            f"PLS1 score (DS ≥ {threshold:g})"
        )

        annotation = (
            rf"$r$ = {r_value:.3f}"
        )

        if spin_results is not None:

            p_spin = spin_results[threshold]

            if p_spin < 0.001:
                annotation += "\n" + r"$P_{\mathrm{spin}} < 0.001$"
            else:
                annotation += (
                    "\n"
                    + rf"$P_{{\mathrm{{spin}}}}$ = {p_spin:.3f}"
                )

        ax.text(
            0.12,
            0.90,
            annotation,
            transform=ax.transAxes,
            fontsize=11
        )

        ax.text(
            -0.18,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main analysis
# ============================================================

def main(args):

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load phenotype
    # --------------------------------------------------------

    mat = loadmat(
        args.phenotype
    )

    if args.phenotype_key not in mat:
        raise KeyError(
            f"Variable '{args.phenotype_key}' "
            "was not found in phenotype file."
        )

    phenotype = np.asarray(
        mat[args.phenotype_key]
    ).squeeze()

    # --------------------------------------------------------
    # 2. Process AHBA expression data
    # --------------------------------------------------------

    expression_donors, counts = (
        abagen.get_expression_data(
            atlas=args.atlas,
            atlas_info=args.atlas_info,
            lr_mirror=True,
            exact=False,
            tolerance=10,
            return_counts=True,
            return_donors=True
        )
    )

    donor_ids, donor_list = (
        prepare_donor_expression(
            expression_donors
        )
    )

    print(
        f"Number of AHBA donors: "
        f"{len(donor_list)}"
    )

    print(
        f"Number of regions: "
        f"{donor_list[0].shape[0]}"
    )

    print(
        f"Number of genes before DS filtering: "
        f"{donor_list[0].shape[1]}"
    )

    if donor_list[0].shape[0] != len(phenotype):
        raise ValueError(
            "Number of atlas regions does not match "
            "the phenotype vector."
        )

    # --------------------------------------------------------
    # 3. Differential stability
    # --------------------------------------------------------

    ds_table = (
        calculate_differential_stability(
            donor_list
        )
    )

    ds_table.to_csv(
        os.path.join(
            args.output_dir,
            "gene_differential_stability.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 4. Group-average expression
    # --------------------------------------------------------

    group_expression = average_donors(
        donor_list
    )

    # --------------------------------------------------------
    # 5. Run PLS under three DS thresholds
    # --------------------------------------------------------

    thresholds = [
        0.0,
        0.1,
        0.2
    ]

    results = {}

    for threshold in thresholds:

        result = run_pls1(
            group_expression,
            phenotype,
            ds_table,
            threshold
        )

        results[threshold] = result

        result["weights"].to_csv(
            os.path.join(
                args.output_dir,
                f"PLS1_gene_weights_DS_{threshold:g}.csv"
            ),
            index=False
        )

    # --------------------------------------------------------
    # 6. Align signs to primary DS >= 0.1 analysis
    # --------------------------------------------------------

    reference = results[0.1]["score"]

    for threshold in [
        0.0,
        0.2
    ]:

        aligned = align_component_sign(
            reference,
            results[threshold]["score"]
        )

        if not np.array_equal(
            aligned,
            results[threshold]["score"]
        ):
            results[threshold]["score"] = aligned
            results[threshold]["weights"][
                "PLS1_weight"
            ] *= -1

    # --------------------------------------------------------
    # 7. Spatial correlations
    # --------------------------------------------------------

    r_ds0 = pearsonr(
        reference,
        results[0.0]["score"]
    )[0]

    r_ds02 = pearsonr(
        reference,
        results[0.2]["score"]
    )[0]

    # --------------------------------------------------------
    # 8. Optional spin tests
    # --------------------------------------------------------

    spin_results = None

    if args.spin_permutations is not None:

        permutations = np.load(
            args.spin_permutations
        )

        _, p0 = spin_test(
            reference,
            results[0.0]["score"],
            permutations
        )

        _, p02 = spin_test(
            reference,
            results[0.2]["score"],
            permutations
        )

        spin_results = {
            0.0: p0,
            0.2: p02
        }

    # --------------------------------------------------------
    # 9. Save regional PLS1 maps
    # --------------------------------------------------------

    score_table = pd.DataFrame({
        "region": group_expression.index,
        "phenotype": phenotype,
        "PLS1_DS_0": results[0.0]["score"],
        "PLS1_DS_0.1": results[0.1]["score"],
        "PLS1_DS_0.2": results[0.2]["score"]
    })

    score_table.to_csv(
        os.path.join(
            args.output_dir,
            "PLS1_regional_scores.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 10. Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({
        "DS_threshold": thresholds,
        "N_genes": [
            results[t]["n_genes"]
            for t in thresholds
        ],
        "PLS1_Y_correlation": [
            results[t]["corr_with_Y"]
            for t in thresholds
        ],
        "Y_variance_explained_percent": [
            results[t][
                "variance_explained_Y"
            ]
            for t in thresholds
        ]
    })

    summary.to_csv(
        os.path.join(
            args.output_dir,
            "PLS_DS_summary.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # 11. Robustness figure
    # --------------------------------------------------------

    plot_robustness(
        main_score=reference,
        alternative_scores={
            0.0: results[0.0]["score"],
            0.2: results[0.2]["score"]
        },
        output_file=os.path.join(
            args.output_dir,
            "PLS1_DS_robustness.png"
        ),
        spin_results=spin_results
    )

    # --------------------------------------------------------
    # 12. Report
    # --------------------------------------------------------

    print("\nPLS1 robustness analysis")
    print("------------------------")

    for threshold in thresholds:

        print(
            f"DS >= {threshold:g}: "
            f"{results[threshold]['n_genes']} genes"
        )

    print(
        f"\nDS >= 0 vs DS >= 0.1: "
        f"r = {r_ds0:.3f}"
    )

    print(
        f"DS >= 0.2 vs DS >= 0.1: "
        f"r = {r_ds02:.3f}"
    )

    if spin_results is not None:

        print(
            f"Pspin (DS >= 0): "
            f"{spin_results[0.0]:.6f}"
        )

        print(
            f"Pspin (DS >= 0.2): "
            f"{spin_results[0.2]:.6f}"
        )


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "PLS transcriptomic analysis with "
            "differential-stability robustness testing."
        )
    )

    parser.add_argument(
        "--atlas",
        required=True,
        help="Path to atlas NIfTI file."
    )

    parser.add_argument(
        "--atlas-info",
        required=True,
        help="Path to atlas information CSV."
    )

    parser.add_argument(
        "--phenotype",
        required=True,
        help="MAT file containing the regional phenotype."
    )

    parser.add_argument(
        "--phenotype-key",
        default="Y",
        help="Variable name in the phenotype MAT file."
    )

    parser.add_argument(
        "--spin-permutations",
        default=None,
        help=(
            "Optional NumPy file containing spin-permutation "
            "indices (n_permutations x n_regions)."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="./results/pls_ds_robustness",
        help="Output directory."
    )

    args = parser.parse_args()

    main(args)