"""
=======================================================================
 Script Name: Subtype_RCCA_Analysis_AgeAdjusted.py

 Purpose:
   This script performs age-adjusted regularized canonical correlation 
   analysis (RCCA) between structural brain network deviations and 
   behavioral/clinical scales across ASD subtypes.

   Age effects are removed from both brain features and behavioral 
   measures before RCCA using linear regression residualization.

=======================================================================
"""


import warnings

import numpy as np
import pandas as pd

from rcca import CCA
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


warnings.filterwarnings("ignore")


# ============================================================
# Step 1: Load data
# ============================================================

df = pd.read_csv(
    "./data/subtype_sf_table_100_new_all.csv"
)

df.columns = [
    col.replace('/', '_')
    for col in df.columns
]


# ============================================================
# Brain features (X)
# ============================================================

columns_to_fit = [
    'Default_Parietal',
    'Default_Anterolateral',
    'Default_Dorsolateral',
    'Default_Retrosplenial',

    'Visual_Lateral',
    'Visual_Dorsal_VentralStream',
    'Visual_V5',
    'Visual_V1',

    'Frontoparietal',
    'DorsalAttention',
    'Premotor_DorsalAttentionII',

    'Language',
    'Salience',
    'CinguloOpercular_Action_mode',

    'MedialParietal',

    'Somatomotor_Hand',
    'Somatomotor_Face',
    'Somatomotor_Foot',

    'Auditory',

    'SomatoCognitiveAction'
]


# ============================================================
# Behavioral measures (Y)
# ============================================================

candidate_ados_full = [

    'ADI_RRB_TOTAL_C',
    'ADOS_STEREO_BEHAV',
    'ADOS_GOTHAM_RRB',

    'SRS_MANNERISMS',

    'VINELAND_PERSONAL_V_SCALED',
    'VINELAND_DOMESTIC_V_SCALED',
    'VINELAND_PLAY_V_SCALED',
    'VINELAND_COPING_V_SCALED',

    'ADOS_TOTAL',
    'ADOS_COMM',
    'ADOS_SOCIAL',
    'ADOS_GOTHAM_SOCAFFECT',

    'ADI_R_SOCIAL_TOTAL_A',
    'ADI_R_VERBAL_TOTAL_BV',

    'SRS_RAW_TOTAL',
    'SRS_AWARENESS',
    'SRS_COGNITION',
    'SRS_COMMUNICATION',
    'SRS_MOTIVATION'
]



# ============================================================
# Confounder
# ============================================================

confounder = "Age"



# ============================================================
# Select ASD clusters
# ============================================================

df = df[
    df['Cluster'].isin([0,1,2])
]


permutations = 1000

rng = np.random.default_rng(42)



results = []

weights_all = []

sig_tracker = {}



# ============================================================
# Function:
# Remove confounding effect
# ============================================================

def regress_out(X, C):

    """
    Remove confounder effects using linear regression.

    X:
        Variables to residualize

    C:
        Confounder(s), e.g. Age
    """

    X_res = np.zeros_like(X)


    for i in range(X.shape[1]):

        model = LinearRegression()

        model.fit(
            C,
            X[:,i]
        )


        X_res[:,i] = (
            X[:,i]
            -
            model.predict(C)
        )


    return X_res




# ============================================================
# Step 2:
# Cluster x behavioral scale RCCA
# ============================================================


for scale in candidate_ados_full:


    sig_tracker[scale] = []


    for cluster in [0,1,2]:


        print(
            "Running:",
            scale,
            "Cluster:",
            cluster
        )


        df_cluster = df[
            df['Cluster']==cluster
        ]



        needed_columns = (
            columns_to_fit
            +
            [scale, confounder]
        )



        if not all(
            col in df_cluster.columns
            for col in needed_columns
        ):

            continue



        temp = df_cluster[
            needed_columns
        ].dropna()



        if temp.shape[0] < 30:

            continue



        # -------------------------
        # Raw variables
        # -------------------------

        X = temp[
            columns_to_fit
        ].values


        Y = temp[
            [scale]
        ].values


        Age = temp[
            [confounder]
        ].values



        # ====================================================
        # Age residualization
        # ====================================================

        X_res = regress_out(
            X,
            Age
        )


        Y_res = regress_out(
            Y,
            Age
        )



        # ====================================================
        # Standardization
        # ====================================================

        scaler_X = StandardScaler()

        scaler_Y = StandardScaler()



        X_std = scaler_X.fit_transform(
            X_res
        )


        Y_std = scaler_Y.fit_transform(
            Y_res
        )



        # ====================================================
        # RCCA
        # ====================================================

        cca = CCA(
            numCC=1,
            reg=0.1
        )


        cca.train(
            [
                X_std,
                Y_std
            ]
        )


        X_c = cca.comps[0]

        Y_c = cca.comps[1]



        r_true, _ = pearsonr(
            X_c[:,0],
            Y_c[:,0]
        )



        # ====================================================
        # Permutation test
        # ====================================================

        r_null=[]



        for _ in range(permutations):


            Y_perm = rng.permutation(
                Y_std
            )


            cca_perm = CCA(
                numCC=1,
                reg=0.1
            )


            cca_perm.train(
                [
                    X_std,
                    Y_perm
                ]
            )



            X_perm_c = cca_perm.comps[0]

            Y_perm_c = cca_perm.comps[1]



            r_perm,_ = pearsonr(
                X_perm_c[:,0],
                Y_perm_c[:,0]
            )


            r_null.append(
                r_perm
            )



        p_val = np.mean(
            np.abs(r_null)
            >=
            np.abs(r_true)
        )



        # ====================================================
        # Save significant results
        # ====================================================


        if p_val < 0.05:


            sig_tracker[scale].append(
                True
            )


            comp_df = pd.DataFrame(
                {
                    "X_c":X_c[:,0],
                    "Y_c":Y_c[:,0]
                }
            )


            comp_df.to_csv(
                f"./subtype/"
                f"cca_age_adjusted_components_cluster"
                f"{cluster}_{scale}.csv",
                index=False
            )


        else:

            sig_tracker[scale].append(
                False
            )



        results.append(
            {
                "Cluster":cluster,
                "Scale":scale,
                "CCA_Component":1,
                "r":round(r_true,4),
                "p":round(p_val,4)
            }
        )



        weights_all.append(
            {
                "Cluster":cluster,
                "Scale":scale,
                "CCA_Component":1,

                **{
                    f"{col}_weight":
                    round(w,4)

                    for col,w in zip(
                        columns_to_fit,
                        cca.ws[0][:,0]
                    )
                }
            }
        )



# ============================================================
# Step 3:
# Keep scales significant in all clusters
# ============================================================


valid_scales = [

    scale

    for scale,sigs

    in sig_tracker.items()

    if sum(sigs)==3

]



results_df = pd.DataFrame(
    results
)


weights_df = pd.DataFrame(
    weights_all
)



results_df = results_df[
    results_df["Scale"].isin(valid_scales)
]


weights_df = weights_df[
    weights_df["Scale"].isin(valid_scales)
]



results_df.to_csv(
    "./subtype/rcca_age_adjusted_significant_results.csv",
    index=False
)


weights_df.to_csv(
    "./subtype/rcca_age_adjusted_weights.csv",
    index=False
)



print(
    "Age-adjusted significant scales:"
)

print(
    results_df["Scale"].unique()
)



# ============================================================
# Step 4:
# Top 5 features
# ============================================================


df_weight = pd.read_csv(
    "./subtype/rcca_age_adjusted_weights.csv",
    index_col=0
)



df_numeric = df_weight.select_dtypes(
    include=[np.number]
)



top_features=[]



for idx,row in df_numeric.iterrows():


    top_indices = (
        row.abs()
        .nlargest(5)
        .index
    )


    result={
        "Index":idx
    }



    for i,col in enumerate(
        top_indices,
        1
    ):

        result[
            f"Feature_{i}"
        ] = col


        result[
            f"Value_{i}"
        ] = row[col]



    top_features.append(
        result
    )



top_df = pd.DataFrame(
    top_features
)



top_df.to_csv(
    "./subtype/rcca_age_adjusted_top5_features.csv",
    index=False
)



print(
    "Finished."
)
