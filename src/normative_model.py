"""
Normative modeling on atlas-based surface area to quantify deviations (HC vs. ASD)
---------------------------------------------------------------------------------

Overview
- Builds a *normative model* of regional cortical surface area from Healthy Controls (HC),
  using age as predictors via a Generalized Additive Model (GAM).
- Applies the HC-trained normative model to ASD subjects to obtain subject-wise,
  region-wise deviation (z) scores.

Data assumptions
- Inputs (CSV): ./data/HC_{index}_combat_100.csv, ./data/ASD_{index}_combat_100.csv
  Required columns:
    - 'Age' (float): subject age
    - 'fd_mean' (float): mean framewise displacement
    - One column per atlas network (e.g., 'Language', 'Visual_V1', ...), values are *surface area*.
- Zeros in network columns are replaced with the nonzero mean of that column (per group).

Modeling
- For each network:
    1) 10-fold CV on HC data with LinearGAM: surface ~ s(Age) + s(FD_mean).
    2) For each fold, compute:
       - HC validation deviation: (observed_HC - predicted_HC) / std(train_HC_surface)
       - ASD deviation: (observed_ASD - predicted_ASD) / std(train_HC_surface)
    3) Average ASD deviations across folds → final ASD deviation vector for that network.

Outputs
- CSV:
    - ./output_100/fig3_HC_{index}_deviation_results.csv
    - ./output_100/fig3_ASD_{index}_deviation_results.csv

Dependencies
- numpy, pandas, matplotlib, scipy, scikit-learn, pygam

Notes
- Deviation is standardized by the *training HC fold* std to reflect atypicality relative to the normative cohort.
- Thresholds are set to ±2.59 (~99% two-tailed) for visualization; adjust as needed.
- Replace slashes in network names with underscores to keep valid column/file names.
"""

import os

os.environ['PYGAME_DISABLE_TQDM'] = '1'  # disable tqdm output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold

index = 'sf'

# === File paths ===
file_path_HC = f'./data/HC_{index}_combat_100.csv'
file_path_ASD = f'./data/ASD_{index}_combat_100.csv'

data_HC = pd.read_csv(file_path_HC)
data_ASD = pd.read_csv(file_path_ASD)

# === Brain networks of interest ===
columns_to_fit = [
    'Default_Parietal', 'Default_Anterolateral',
    'Default_Dorsolateral', 'Default_Retrosplenial', 'Visual_Lateral',
    'Visual_Dorsal_VentralStream', 'Visual_V5', 'Visual_V1',
    'Frontoparietal', 'DorsalAttention', 'Premotor_DorsalAttentionII',
    'Language', 'Salience', 'CinguloOpercular_Action_mode',
    'MedialParietal', 'Somatomotor_Hand', 'Somatomotor_Face',
    'Somatomotor_Foot', 'Auditory', 'SomatoCognitiveAction'
]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
deviation_df_asd = pd.DataFrame()
deviation_df_hc = pd.DataFrame()

for column in columns_to_fit:
    # --- Extract HC data ---
    age_HC = data_HC['Age'].values
    surface_HC = data_HC[column].values
    fd_HC = data_HC['fd_mean'].values
    surface_HC[surface_HC == 0] = np.nanmean(surface_HC[surface_HC != 0])  # replace zeros

    # --- Extract ASD data ---
    age_ASD = data_ASD['Age'].values
    surface_ASD = data_ASD[column].values
    fd_ASD = data_ASD['fd_mean'].values
    surface_ASD[surface_ASD == 0] = np.nanmean(surface_ASD[surface_ASD != 0])  # replace zeros

    # Initialize deviation arrays
    hc_deviation = np.zeros(len(age_HC))
    asd_deviation_accum = np.zeros((len(age_ASD), kf.get_n_splits()))

    # --- Cross-validation within HC group ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(age_HC)):
        age_train, fd_train, surf_train = age_HC[train_idx], fd_HC[train_idx], surface_HC[train_idx]
        gam = LinearGAM(s(0) + s(1)).gridsearch(np.column_stack([age_train, fd_train]), surf_train)

        # HC validation set
        age_val, fd_val, surf_val = age_HC[val_idx], fd_HC[val_idx], surface_HC[val_idx]
        pred_val = gam.predict(np.column_stack([age_val, fd_val]))
        deviation_val = (surf_val - pred_val) / np.std(surf_train)
        hc_deviation[val_idx] = deviation_val

        # Apply GAM model to ASD group
        pred_asd = gam.predict(np.column_stack([age_ASD, fd_ASD]))
        deviation_asd = (surface_ASD - pred_asd) / np.std(surf_train)
        asd_deviation_accum[:, fold] = deviation_asd

    # Mean ASD deviation across folds
    asd_deviation_mean = np.mean(asd_deviation_accum, axis=1)

    deviation_df_hc[column] = hc_deviation
    deviation_df_asd[column] = asd_deviation_mean

    # --- Visualization ---
    surface_pred_HC = gam.predict(np.column_stack([
        np.linspace(np.min(age_HC), np.max(age_HC), 100),
        np.full(100, np.mean(fd_HC))
    ]))
    age_pred = np.linspace(np.min(age_HC), np.max(age_HC), 100)
    pos_threshold, neg_threshold = 2.59, -2.59

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (1) GAM fit with ASD scatter
    ax1 = axes[0]
    ax1.scatter(age_ASD, surface_ASD, color='#659561', alpha=0.6, label='ASD Data')
    ax1.plot(age_pred, surface_pred_HC, color='#fe8a77', linewidth=2, label='GAM Fit')
    upper_threshold = surface_pred_HC + pos_threshold * np.std(surf_train)
    lower_threshold = surface_pred_HC + neg_threshold * np.std(surf_train)
    ax1.plot(age_pred, upper_threshold, color='#fe8a77', linestyle='--', linewidth=1.5)
    ax1.plot(age_pred, lower_threshold, color='#fe8a77', linestyle='--', linewidth=1.5, label='z-threshold')
    ax1.set_title(f'{column} vs Age')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Surface Area')
    ax1.legend()

    # (2) ASD deviation scores
    deviation_z = asd_deviation_mean
    sorted_indices = np.argsort(deviation_z)
    sorted_deviation_z = deviation_z[sorted_indices]
    ax2 = axes[1]
    ax2.barh(range(len(sorted_deviation_z)), sorted_deviation_z, color='#cd6f5b', alpha=0.7, edgecolor='none')
    ax2.axvline(x=pos_threshold, color='#ab96e9', linestyle='--', label=f'+ threshold = {pos_threshold:.2f}')
    ax2.axvline(x=neg_threshold, color='#ab96e9', linestyle='--', label=f'- threshold = {neg_threshold:.2f}')
    ax2.axvline(x=0, color='#8377a8', linestyle='--')
    ax2.set_title('ASD Deviation Scores')
    ax2.set_xlabel('Deviation (z)')
    ax2.set_ylabel('Subjects (sorted)')
    ax2.legend()

    # (3) HC vs ASD deviation comparison
    hc_deviation_z = hc_deviation
    asd_deviation_z = deviation_z
    violin_data = [hc_deviation_z, asd_deviation_z]
    violin_labels = ['HC', 'ASD']
    scatter_colors = ['#899FB0', '#cd6f5b']

    t_stat, p_value = ttest_ind(surface_HC, surface_ASD)

    ax3 = axes[2]
    for i, (data, color) in enumerate(zip(violin_data, scatter_colors)):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax3.scatter(x, data, color=color, alpha=0.6, label=violin_labels[i])
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        ax3.plot([i + 1 - 0.2, i + 1 + 0.2], [median, median], color='black', linestyle='--', lw=1)
        ax3.plot([i + 1 - 0.1, i + 1 + 0.1], [q1, q1], color='black', linestyle='--', lw=1)
        ax3.plot([i + 1 - 0.1, i + 1 + 0.1], [q3, q3], color='black', linestyle='--', lw=1)

    ax3.axhline(y=pos_threshold, color='#a0bfa0', linestyle='--', label='z-threshold')
    ax3.axhline(y=neg_threshold, color='#a0bfa0', linestyle='--')
    ax3.axhline(y=0, color='#918ac2', linestyle='--')
    ax3.set_title(f'Deviation Scores (HC vs ASD)\n(p={p_value:.3f})')
    ax3.set_ylabel('Deviation (z)')
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(violin_labels)
    ax3.legend()

    plt.tight_layout()
    plt.show()

# --- Save results ---
deviation_df_hc['Age'] = data_HC['Age'].values
deviation_df_hc['FD_mean'] = data_HC['fd_mean'].values
deviation_df_asd['Age'] = data_ASD['Age'].values
deviation_df_asd['FD_mean'] = data_ASD['fd_mean'].values

deviation_df_hc.to_csv(f'./output_100/fig3_HC_{index}_deviation_results.csv', index=False)
deviation_df_asd.to_csv(f'./output_100/fig3_ASD_{index}_deviation_results.csv', index=False)

print("Deviation scores have been saved:")
print("  - HC_brain_networks_deviation_results.csv")
print("  - ASD_brain_networks_deviation_results.csv")
