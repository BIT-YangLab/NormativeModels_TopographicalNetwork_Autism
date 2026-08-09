#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot kernel density distributions of model-fit metrics.

The input CSV should contain the following columns:
    SMSE
    MSLL

Outputs
-------
SMSE_MSLL_density.png
SMSE_MSLL_density.pdf
SMSE_MSLL_density.svg
"""

from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# =============================================================================
# Figure settings
# =============================================================================

SHOW_HISTOGRAM = False


def configure_matplotlib():
    """Configure publication-quality figure settings."""

    available_fonts = {
        font.name for font in fm.fontManager.ttflist
    }

    if "Arial" in available_fonts:
        font_family = "Arial"
    else:
        font_family = "Liberation Sans"
        warnings.warn(
            "Arial was not detected. "
            "Liberation Sans will be used instead."
        )

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# =============================================================================
# Utility functions
# =============================================================================

def calculate_curve_area(y, x):
    """Calculate area under the curve using trapezoidal integration."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def load_metrics(input_file):
    """
    Load model-fit metrics from a CSV file.

    Required columns
    ----------------
    SMSE
    MSLL
    """

    data = pd.read_csv(input_file)

    required_columns = ["SMSE", "MSLL"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    smse = data["SMSE"].to_numpy(dtype=float)
    msll = data["MSLL"].to_numpy(dtype=float)

    return smse, msll


# =============================================================================
# KDE plotting
# =============================================================================

def draw_kde(
    ax,
    data,
    x_limits,
    y_limits,
    x_ticks,
    y_ticks,
    x_label,
    panel_label,
):
    """
    Plot a one-dimensional Gaussian kernel density distribution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target plotting axis.

    data : array-like
        Input observations.

    x_limits : tuple
        Limits of the x-axis.

    y_limits : tuple
        Limits of the y-axis.

    x_ticks : sequence
        Tick locations on the x-axis.

    y_ticks : sequence
        Tick locations on the y-axis.

    x_label : str
        Label of the x-axis.

    panel_label : str
        Panel identifier.
    """

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size < 2:
        raise ValueError(
            f"{x_label} contains fewer than two valid observations."
        )

    x_grid = np.linspace(
        x_limits[0],
        x_limits[1],
        2000,
    )

    kde = gaussian_kde(
        data,
        bw_method="scott",
    )

    density = kde(x_grid)

    if SHOW_HISTOGRAM:
        ax.hist(
            data,
            bins="auto",
            density=True,
            alpha=0.25,
            edgecolor="none",
        )

    ax.fill_between(
        x_grid,
        density,
        0,
        color="0.82",
        linewidth=0,
    )

    ax.plot(
        x_grid,
        density,
        color="0.65",
        linewidth=1.0,
    )

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")

    ax.text(
        -0.22,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
    )

    kde_area = calculate_curve_area(
        density,
        x_grid,
    )

    print(f"\n{x_label}")
    print("-" * 45)
    print(f"N                  = {data.size}")
    print(f"Mean               = {np.mean(data):.6f}")
    print(f"Standard deviation = {np.std(data, ddof=1):.6f}")
    print(f"KDE area           = {kde_area:.6f}")


# =============================================================================
# Main analysis
# =============================================================================

def main(args):

    configure_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    smse, msll = load_metrics(
        args.input_file
    )

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.2, 3.35),
    )

    # Panel A: SMSE
    draw_kde(
        ax=axes[0],
        data=smse,
        x_limits=tuple(args.smse_xlim),
        y_limits=tuple(args.smse_ylim),
        x_ticks=np.linspace(
            args.smse_xlim[0],
            args.smse_xlim[1],
            4,
        ),
        y_ticks=np.linspace(
            args.smse_ylim[0],
            args.smse_ylim[1],
            6,
        ),
        x_label="SMSE",
        panel_label="A",
    )

    # Panel B: MSLL
    draw_kde(
        ax=axes[1],
        data=msll,
        x_limits=tuple(args.msll_xlim),
        y_limits=tuple(args.msll_ylim),
        x_ticks=np.linspace(
            args.msll_xlim[0],
            args.msll_xlim[1],
            4,
        ),
        y_ticks=np.linspace(
            args.msll_ylim[0],
            args.msll_ylim[1],
            6,
        ),
        x_label="MSLL",
        panel_label="B",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.20,
        top=0.94,
        wspace=0.34,
    )

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    output_base = (
        output_dir
        / "SMSE_MSLL_density"
    )

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")

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

    print("\nFigure saved:")
    print(png_path)
    print(pdf_path)
    print(svg_path)


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Plot kernel density distributions "
            "of SMSE and MSLL."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help=(
            "CSV file containing SMSE and MSLL columns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="./density_results",
        help="Directory for output figures.",
    )

    parser.add_argument(
        "--smse-xlim",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="X-axis limits for SMSE.",
    )

    parser.add_argument(
        "--smse-ylim",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="Y-axis limits for SMSE density.",
    )

    parser.add_argument(
        "--msll-xlim",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="X-axis limits for MSLL.",
    )

    parser.add_argument(
        "--msll-ylim",
        nargs=2,
        type=float,
        required=True,
        metavar=("MIN", "MAX"),
        help="Y-axis limits for MSLL density.",
    )

    args = parser.parse_args()

    main(args)