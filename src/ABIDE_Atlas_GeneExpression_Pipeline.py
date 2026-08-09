"""
=======================================================================
 Script Name: ABIDE_Atlas_GeneExpression_Pipeline.py

 Purpose:
   This script provides a two-step pipeline for preparing region-wise 
   gene expression data aligned to an ABIDE atlas:

   Step 1. Atlas Consistency Check
       - Verify that the ABIDE atlas (NIfTI file) and the annotation 
         table (CSV) are consistent.
       - Ensure that all region IDs match between the files.

   Step 2. Gene Expression Extraction
       - Use the `abagen` toolbox to extract gene expression profiles 
         from the Allen Human Brain Atlas (AHBA).
       - Map gene expression data to atlas-defined regions of interest (ROIs).
       - Perform quality control (check missing values, impute NaNs).
       - Save processed expression data and sample counts.

 Inputs:
   - ABIDE_atlas.nii.gz
       Brain atlas in NIfTI format defining ROIs.
   - ABIDE_atlas.csv
       Annotation table with atlas metadata (ROI ID, hemisphere, structure).

 Outputs:
   - Console log: Atlas validation report (volume shape, unique labels, ID mismatches).
   - ABIDE_gene_expression_data.csv: ROI × Gene expression matrix (NaNs imputed).
   - ABIDE_sample_counts.csv: Sample counts per ROI.

 Dependencies:
   - abagen
   - nibabel
   - numpy
   - pandas
=======================================================================
"""

import abagen
import nibabel as nib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Step 1. Atlas Consistency Check
# ----------------------------------------------------------------------
atlas_image = './ABIDE_atlas.nii.gz'
atlas_info_csv = './ABIDE_atlas.csv'

# Load atlas image
atlas_img = nib.load(atlas_image)
atlas_data = atlas_img.get_fdata()
atlas_shape = atlas_data.shape
print(f"[Atlas Check] Atlas volume shape: {atlas_shape}")

# Extract unique region labels
unique_labels = np.unique(atlas_data)
print(f"[Atlas Check] Atlas contains {len(unique_labels)} unique labels:")
print(unique_labels)

# Load atlas annotation table
atlas_info = pd.read_csv(atlas_info_csv)

# Check required columns
required_columns = {"id", "hemisphere", "structure"}
if not required_columns.issubset(set(atlas_info.columns)):
    raise ValueError(f" Error: `atlas_info.csv` missing columns {required_columns - set(atlas_info.columns)}")

# Verify IDs match between atlas and CSV
atlas_ids = set(atlas_info["id"])
unique_labels_set = set(unique_labels)

missing_ids = unique_labels_set - atlas_ids   # In atlas but not in CSV
extra_ids   = atlas_ids - unique_labels_set   # In CSV but not in atlas

if missing_ids:
    print(f" {len(missing_ids)} labels in atlas not found in CSV:")
    print(missing_ids)
if extra_ids:
    print(f" {len(extra_ids)} labels in CSV not found in atlas:")
    print(extra_ids)

print(" Atlas consistency check complete!")

# ----------------------------------------------------------------------
# Step 2. Gene Expression Extraction
# ----------------------------------------------------------------------
print("\n[Gene Expression] Extracting data with abagen...")

expression_data, counts = abagen.get_expression_data(
    atlas=atlas_image,
    atlas_info=atlas_info_csv,
    lr_mirror=True,
    exact=False,     # Allow approximate ROI matches
    tolerance=10,    # Search radius (mm), larger than default (5 mm)
    return_counts=True
)

# Inspect outputs
print(expression_data.head())
print(counts.head())

# Check and handle missing values
n_missing = np.isnan(expression_data).sum().sum()
print(f"[Gene Expression] Number of missing values: {n_missing}")
expression_data.fillna(expression_data.mean(), inplace=True)

# Save processed data
expression_data.to_csv("./ABIDE_gene_expression_data.csv")
counts.to_csv("./ABIDE_sample_counts.csv")

print(" Gene expression data and sample counts saved successfully!")
