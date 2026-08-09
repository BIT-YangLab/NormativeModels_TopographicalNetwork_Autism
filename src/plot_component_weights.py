#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot component weights as a bar chart.

The script reads component weights from a CSV file, selects one row,
and highlights the features with the largest absolute weights.

Outputs
-------
component_weights.png
component_weights.pdf
component_weights.svg
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Data loading
# =============================================================================

def load_component_weights(input_file, row_index=0):
    """
    Load component weights from a CSV file.

    Parameters
    ----------
    input_file : str or Path
        Path to the input CSV file.

    row_index : int
        Row containing the component weights.

    Returns
    -------
    feature_names : list of str
        Names of features.

    weights : ndarray
        Component weights.
    """

    df = pd.read_csv(input_file)

    if row_index < 0 or row_index >= len(df):
        raise IndexError(
            f"row_index={row_index} is outside the valid range."
        )

    # Retain numeric columns only.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric weight columns were found in the input file."
        )

    row = df.iloc[row_index]

    feature_names = numeric_columns

    weights = row[
        numeric_columns
    ].to_numpy(dtype=float)

    return feature_names, weights


# =============================================================================
# Plotting
# =============================================================================

def plot_component_weights(
    feature_names,
    weights,
    top_n,
    output_dir,
    output_name,
):
    """
    Plot component weights and highlight features with the largest
    absolute weights.

    Parameters
    ----------
    feature_names : sequence of str
        Feature names shown along the x-axis.

    weights : array-like
        Component weights.

    top_n : int
        Number of largest absolute weights to highlight.

    output_dir : str or Path
        Directory for output figures.

    output_name : str
        Base name of the output files.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if len(feature_names) != len(weights):
        raise ValueError(
            "The number of feature names does not match "
            "the number of weights."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    top_n = min(
        top_n,
        len(weights),
    )

    # -------------------------------------------------------------------------
    # Identify features with the largest absolute weights
    # -------------------------------------------------------------------------

    top_indices = np.argsort(
        np.abs(weights)
    )[-top_n:]

    highlight_mask = np.zeros(
        len(weights),
        dtype=bool,
    )

    highlight_mask[
        top_indices
    ] = True

    colors = [
        "gray" if highlight
        else "tab:blue"
        for highlight in highlight_mask
    ]

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    x = np.arange(
        len(weights)
    )

    ax.bar(
        x,
        weights,
        color=colors,
        width=0.8,
    )

    # -------------------------------------------------------------------------
    # Axis formatting
    # -------------------------------------------------------------------------

    ax.set_xticks(x)

    # Keep tick marks but remove tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=6,
        width=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Zero-reference line.
    ax.axhline(
        y=0,
        linewidth=0.8,
        color="black",
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_base = (
        output_dir
        / output_name
    )

    png_path = output_base.with_suffix(
        ".png"
    )

    pdf_path = output_base.with_suffix(
        ".pdf"
    )

    svg_path = output_base.with_suffix(
        ".svg"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("Figure saved:")
    print(png_path)
    print(pdf_path)
    print(svg_path)


# =============================================================================
# Main
# =============================================================================

def main(args):
    """Run component-weight visualization."""

    feature_names, weights = load_component_weights(
        input_file=args.input_file,
        row_index=args.row_index,
    )

    print(
        f"Number of features: {len(weights)}"
    )

    plot_component_weights(
        feature_names=feature_names,
        weights=weights,
        top_n=args.top_n,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Visualize component weights and highlight "
            "features with the largest absolute weights."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="CSV file containing component weights.",
    )

    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help=(
            "Row containing the component weights "
            "(default: first row)."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help=(
            "Number of largest absolute weights to highlight."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="./figures",
        help="Directory for output figures.",
    )

    parser.add_argument(
        "--output-name",
        default="component_weights",
        help="Base name of the output figure.",
    )

    args = parser.parse_args()

    main(args)