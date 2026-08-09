"""
Coefficient of Variation (CV) analysis and GLM fitting for HC vs ASD groups
--------------------------------------------------------------------------

Overview
1. Compute the coefficient of variation (CV = std/mean × 100) for each brain network
   across subjects in HC and ASD groups.
2. Compare CV values across groups by merging them with group-level mean values.
3. Fit Generalized Linear Models (GLMs) separately for HC and ASD to assess the
   relationship between group means and variability (CV).
4. Save predicted CV values and residuals for further analysis.

Inputs
- hc_mode.csv : subject-level data for HC (first 20 columns = networks)
- asd_mode.csv: subject-level data for ASD (first 20 columns = networks)

Outputs
- cv_comparison.csv : table of raw CV values (%), HC vs ASD
- cv_glm_results_all.csv : merged table containing group means, observed CV,
                           predicted CV (from GLM), and residuals

Dependencies
- pandas
- statsmodels

Notes
- GLM family is Gaussian with identity link, equivalent to linear regression.
- Results can be used to check whether variability (CV) scales with group mean levels.
"""

import pandas as pd

# === Step 1: Compute CV for HC and ASD ===
# Load subject-level data
hc_df = pd.read_csv('hc_mode.csv')
asd_df = pd.read_csv('asd_mode.csv')

# Use first 20 columns (brain networks)
hc_data = hc_df.iloc[:, 0:20]
asd_data = asd_df.iloc[:, 0:20]

# Define CV function
def compute_cv(df):
    return (df.std() / df.mean()) * 100

# Compute CV per column
hc_cv = compute_cv(hc_data)
asd_cv = compute_cv(asd_data)

# Combine into DataFrame
cv_comparison = pd.DataFrame({
    'HC_CV (%)': hc_cv,
    'ASD_CV (%)': asd_cv
})

print(cv_comparison)
cv_comparison.to_csv('cv_comparison.csv')


# === Step 2: GLM analysis (CV vs group mean) ===
import statsmodels.api as sm

# Load CV comparison and group means
cv_df = pd.read_csv('cv_comparison.csv', index_col=0)
group_mean_df = pd.read_excel('sf_group.xlsx', index_col=0)

# Merge CV with group means
df = cv_df.join(group_mean_df)

# --- HC ---
X_hc = sm.add_constant(df[['HC']])     # predictor = group mean (HC)
y_hc = df['HC_CV (%)']                 # response = CV (%)
model_hc = sm.GLM(y_hc, X_hc, family=sm.families.Gaussian()).fit()
df['HC_CV_predicted'] = model_hc.fittedvalues
df['HC_CV_residual'] = model_hc.resid_response

# --- ASD ---
X_asd = sm.add_constant(df[['ASD']])
y_asd = df['ASD_CV (%)']
model_asd = sm.GLM(y_asd, X_asd, family=sm.families.Gaussian()).fit()
df['ASD_CV_predicted'] = model_asd.fittedvalues
df['ASD_CV_residual'] = model_asd.resid_response

# Save merged results
df.to_csv('cv_glm_results_all.csv')

# Display a preview
print(df[['HC_CV (%)', 'HC_CV_predicted', 'HC_CV_residual']].head())
print(df[['ASD_CV (%)', 'ASD_CV_predicted', 'ASD_CV_residual']].head())

