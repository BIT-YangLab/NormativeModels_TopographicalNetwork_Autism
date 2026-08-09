"""
Group comparison of deviation scores (HC vs ASD) across brain networks
---------------------------------------------------------------------

Overview
- Loads deviation (z) scores computed previously from normative modeling
  (see fig3_HC_sf_deviation_results.csv and fig3_ASD_sf_deviation_results.csv).
- Performs independent two-sample t-tests for each atlas-defined network
  to test for group differences between ASD and HC.
- Visualizes results as a bar plot of -log10(p-values) with significance markers.

Data assumptions
- Input files:
    ./output_100/fig3_HC_sf_deviation_results.csv
    ./output_100/fig3_ASD_sf_deviation_results.csv

Statistical testing
- Test: two-sample t-test (ASD vs HC) for each region
- Significance thresholds:
    * p < 0.05 → marked with '*'
    * p < 0.01 → marked with '**'
- Visualization: -log10(p-values) for each region, with threshold lines

Dependencies
- numpy, pandas, scipy, matplotlib

"""

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np


columns_to_fit = [
    'Default_Parietal', 'Default_Anterolateral',
    'Default_Dorsolateral', 'Default_Retrosplenial', 'Visual_Lateral',
    'Visual_Dorsal/VentralStream', 'Visual_V5', 'Visual_V1',
    'Frontoparietal', 'DorsalAttention', 'Premotor/DorsalAttentionII',
    'Language', 'Salience', 'CinguloOpercular/Action_mode',
    'MedialParietal', 'Somatomotor_Hand', 'Somatomotor_Face',
    'Somatomotor_Foot', 'Auditory', 'SomatoCognitiveAction'
]

# Replace invalid characters for file-safe column names
columns_to_fit = [col.replace('/', '_') for col in columns_to_fit]

# === Load HC and ASD deviation score data ===
hc_df = pd.read_csv('./output_100/fig3_HC_sf_deviation_results.csv')
asd_df = pd.read_csv('./output_100/fig3_ASD_sf_deviation_results.csv')

# === Statistical testing: two-sample t-test for each region ===
results = {}
for col in columns_to_fit:
    hc_data = hc_df[col]
    asd_data = asd_df[col]
    t_stat, p_val = ttest_ind(asd_data, hc_data)
    results[col] = (t_stat, p_val)

# === Visualization: bar plot of -log10(p-value) ===
neg_log_p = [-np.log10(p_val) for _, p_val in results.values()]

# Mark significance: ** (p<0.01), * (p<0.05)
significance = ['**' if p < 0.01 else '*' if p < 0.05 else '' for _, p in results.values()]

plt.figure(figsize=(14, 6))
bars = plt.bar(columns_to_fit, neg_log_p, color='skyblue')

# Add reference lines for significance thresholds
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
plt.axhline(-np.log10(0.01), color='orange', linestyle='--', label='p = 0.01')

# Annotate bars with significance markers
for i, (bar, sig) in enumerate(zip(bars, significance)):
    if sig:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 sig, ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=90)
plt.ylabel('-log10(p-value)')
plt.title('Statistical Difference Between ASD and HC by Region')
plt.legend()
plt.tight_layout()
plt.show()

# === Save results table ===
# Export full results (t-statistic, p-value, -log10(p))
full_results_df = pd.DataFrame.from_dict(results, orient='index', columns=['t_statistic', 'p_value'])
full_results_df['-log10(p)'] = -np.log10(full_results_df['p_value'])
full_results_df.to_csv('./output_100/fig3a_ttest_results.csv')
