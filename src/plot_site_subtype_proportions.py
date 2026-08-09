#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot site-wise subtype proportions as vertical stacked bar charts.

Two figures are generated:
1. Full-sample subtype proportions across imaging sites.
2. Leave-one-site-out (LOSO) subtype proportions across imaging sites.

Required input files
--------------------
Full sample:
    01_fullsample_site_proportions.csv
    01_fullsample_site_counts.csv

LOSO:
    02_loso_site_proportions.csv
    02_loso_site_counts.csv

Outputs
-------
    01_fullsample_site_proportions_vertical.png
    01_fullsample_site_proportions_vertical.pdf
    02_loso_site_proportions_vertical.png
    02_loso_site_proportions_vertical.pdf

The script automatically detects site and subtype columns, converts subtype
proportions from [0, 1] to percentages when necessary, and orders sites using
the predefined site order below.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

SUBTYPE_NAMES = ["Subtype 1", "Subtype 2", "Subtype 3"]
SUBTYPE_COLORS = [
    "#339684",  # Subtype 1
    "#456990",  # Subtype 2
    "#EF767A",  # Subtype 3
]

SITE_NAME_MAP = {
    1: "BNI",
    2: "EMC",
    3: "ETHZ",
    4: "GU",
    5: "IP",
    6: "IU",
    7: "UCD",
    8: "CMU",
    9: "Caltech",
    10: "KKI",
    11: "Leuven",
    12: "MaxMun",
    13: "NYU",
    14: "OHSU",
    15: "Olin",
    16: "Pitt",
    17: "SBL",
    18: "SDSU",
    19: "Stanford",
    20: "Trinity",
    21: "UCLA",
    22: "UM",
    23: "UPSM",
    24: "USM",
    25: "Yale",
}

SITE_ORDER = [SITE_NAME_MAP[i] for i in sorted(SITE_NAME_MAP)]

FIG_WIDTH_MM = 127
FIG_HEIGHT_MM = 75
MM_TO_INCH = 1.0 / 25.4


# =============================================================================
# I/O and table utilities
# =============================================================================

def read_csv_with_possible_index(path: Path) -> pd.DataFrame:
    """Read a CSV file and repair an unnamed first index column if present."""
    df = pd.read_csv(path)

    if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Site"})

    return df


def detect_site_column(df: pd.DataFrame) -> str:
    """Detect the column containing imaging-site labels."""
    preferred_names = ["Site", "site", "SITE", "HeldOutSite"]

    for column in preferred_names:
        if column in df.columns:
            return column

    for column in df.columns:
        if "site" in str(column).lower():
            return column

    return str(df.columns[0])


def extract_number_for_sort(value: object) -> int:
    """Extract the last integer in a label for natural subtype ordering."""
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[-1]) if numbers else 9999


def detect_subtype_columns(df: pd.DataFrame) -> list[str]:
    """Detect the three subtype/cluster columns in a table."""
    candidate_columns = [
        column
        for column in df.columns
        if "site" not in str(column).lower()
    ]

    named_columns = [
        column
        for column in candidate_columns
        if "cluster" in str(column).lower()
        or "subtype" in str(column).lower()
    ]

    if len(named_columns) >= 3:
        return sorted(named_columns, key=extract_number_for_sort)[:3]

    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if len(numeric_columns) >= 3:
        return sorted(numeric_columns, key=extract_number_for_sort)[:3]

    raise ValueError("Could not automatically detect three subtype columns.")


def standardize_subtype_names(
    df: pd.DataFrame,
    subtype_columns: Sequence[str],
) -> pd.DataFrame:
    """Rename subtype columns to 'Subtype 1', 'Subtype 2', and 'Subtype 3'."""
    subtype_columns = sorted(subtype_columns, key=extract_number_for_sort)

    rename_map = {
        subtype_columns[i]: SUBTYPE_NAMES[i]
        for i in range(3)
    }

    return df.rename(columns=rename_map)


def convert_site_to_name(value: object) -> str:
    """Convert numeric or textual site labels to standardized site names."""
    text = str(value).strip()

    numbers = re.findall(r"\d+", text)
    if numbers:
        site_index = int(numbers[-1])
        if site_index in SITE_NAME_MAP:
            return SITE_NAME_MAP[site_index]

    for site_name in SITE_ORDER:
        if text.lower() == site_name.lower():
            return site_name

    return text


def prepare_site_distribution_tables(
    proportion_file: Path,
    count_file: Path,
) -> pd.DataFrame:
    """Read, standardize, merge, and order site-level subtype tables."""
    proportion_df = read_csv_with_possible_index(proportion_file)
    count_df = read_csv_with_possible_index(count_file)

    proportion_site_column = detect_site_column(proportion_df)
    count_site_column = detect_site_column(count_df)

    proportion_df = standardize_subtype_names(
        proportion_df,
        detect_subtype_columns(proportion_df),
    )
    count_df = standardize_subtype_names(
        count_df,
        detect_subtype_columns(count_df),
    )

    proportion_df = proportion_df.rename(
        columns={proportion_site_column: "Site"}
    )
    count_df = count_df.rename(columns={count_site_column: "Site"})

    proportion_df = proportion_df[["Site", *SUBTYPE_NAMES]].copy()
    count_df = count_df[["Site", *SUBTYPE_NAMES]].copy()

    proportion_df["Site"] = proportion_df["Site"].apply(convert_site_to_name)
    count_df["Site"] = count_df["Site"].apply(convert_site_to_name)

    merged_df = pd.merge(
        proportion_df,
        count_df,
        on="Site",
        suffixes=("_prop", "_count"),
    )

    merged_df["N"] = sum(
        merged_df[f"{subtype}_count"]
        for subtype in SUBTYPE_NAMES
    )

    # Convert proportions in [0, 1] to percentages.
    for subtype in SUBTYPE_NAMES:
        column = f"{subtype}_prop"
        if merged_df[column].max() <= 1.0 + 1e-8:
            merged_df[column] *= 100.0

    merged_df["Site"] = pd.Categorical(
        merged_df["Site"],
        categories=SITE_ORDER,
        ordered=True,
    )

    return merged_df.sort_values("Site").reset_index(drop=True)


# =============================================================================
# Plotting
# =============================================================================

def configure_matplotlib() -> None:
    """Set figure-wide typography and vector-output settings."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_site_subtype_proportions(
    data: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    panel_letter: str,
    title: str,
) -> None:
    """Draw a vertical stacked bar chart of subtype proportions by site."""
    configure_matplotlib()

    n_sites = len(data)
    x_positions = np.arange(n_sites)
    bottom = np.zeros(n_sites)

    figure_size = (
        FIG_WIDTH_MM * MM_TO_INCH,
        FIG_HEIGHT_MM * MM_TO_INCH,
    )
    fig, ax = plt.subplots(figsize=figure_size)

    for subtype_index, subtype_name in enumerate(SUBTYPE_NAMES):
        values = data[f"{subtype_name}_prop"].to_numpy(dtype=float)

        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            width=0.78,
            color=SUBTYPE_COLORS[subtype_index],
            edgecolor="white",
            linewidth=0.5,
            label=subtype_name,
            zorder=2,
        )

        for site_index, value in enumerate(values):
            if value >= 8:
                ax.text(
                    x_positions[site_index],
                    bottom[site_index] + value / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                    fontweight="bold",
                    zorder=3,
                )

        bottom += values

    site_labels = [
        f"{site}\n(n={int(n)})"
        for site, n in zip(data["Site"], data["N"])
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        site_labels,
        rotation=45,
        ha="right",
        fontweight="bold",
    )
    ax.set_xlim(-0.5, n_sites - 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.set_xlabel("Imaging site", fontweight="bold")
    ax.set_ylabel("Subjects (%)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
        zorder=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.text(
        0.012,
        0.985,
        panel_letter,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, facecolor="white")
    fig.savefig(output_pdf, facecolor="white")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def run_analysis(result_dir: Path) -> None:
    """Generate full-sample and LOSO site-by-subtype figures."""
    full_proportion_file = result_dir / "01_fullsample_site_proportions.csv"
    full_count_file = result_dir / "01_fullsample_site_counts.csv"
    loso_proportion_file = result_dir / "02_loso_site_proportions.csv"
    loso_count_file = result_dir / "02_loso_site_counts.csv"

    full_data = prepare_site_distribution_tables(
        full_proportion_file,
        full_count_file,
    )
    plot_site_subtype_proportions(
        data=full_data,
        output_png=result_dir / "01_fullsample_site_proportions_vertical.png",
        output_pdf=result_dir / "01_fullsample_site_proportions_vertical.pdf",
        panel_letter="a",
        title="Full-sample subtype proportions across sites",
    )

    loso_data = prepare_site_distribution_tables(
        loso_proportion_file,
        loso_count_file,
    )
    plot_site_subtype_proportions(
        data=loso_data,
        output_png=result_dir / "02_loso_site_proportions_vertical.png",
        output_pdf=result_dir / "02_loso_site_proportions_vertical.pdf",
        panel_letter="b",
        title="LOSO subtype proportions across sites",
    )

    print("Finished.")
    print(f"Results saved to: {result_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot full-sample and LOSO site-wise subtype proportions "
            "as vertical stacked bar charts."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory containing the four input CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_analysis(args.result_dir)


if __name__ == "__main__":
    main()
