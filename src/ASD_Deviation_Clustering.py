"""
=======================================================================
 Script Name: ASD_Deviation_Clustering.py

 Purpose:
   This script performs clustering analysis on ASD individuals based on 
   deviations in structural brain networks. The pipeline includes data 
   preprocessing, feature standardization, unsupervised clustering, 
   dimensionality reduction for visualization, and cluster-wise profiling.

 Inputs:
   - fig3_ASD_sf_deviation_results_new.csv
       A CSV file containing deviation values across 20 predefined 
       structural brain networks for ASD individuals.
   
   Columns of interest include:
       ['Default_Parietal', 'Default_Anterolateral', 'Default_Dorsolateral',
        'Default_Retrosplenial', 'Visual_Lateral', 'Visual_Dorsal_VentralStream',
        'Visual_V5', 'Visual_V1', 'Frontoparietal', 'DorsalAttention',
        'Premotor_DorsalAttentionII', 'Language', 'Salience',
        'CinguloOpercular_Action_mode', 'MedialParietal', 'Somatomotor_Hand',
        'Somatomotor_Face', 'Somatomotor_Foot', 'Auditory', 
        'SomatoCognitiveAction']

 Outputs:
   - Scatter plot: PCA visualization of clustering results
   - Heatmap: Mean deviation per cluster across structural networks
   - Console outputs:
       • Cluster labels for each subject
       • Mean deviation profiles for each cluster
       • One-hot encoded cluster labels

 Dependencies:
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn
=======================================================================
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the CSV file network deviation results
df = pd.read_csv("./output_100/fig3_ASD_sf_deviation_results_new.csv")

# Replace illegal characters "/" in column names with "_"
df.columns = [col.replace('/', '_') for col in df.columns]

columns_to_fit = [
    'Default_Parietal', 'Default_Anterolateral', 'Default_Dorsolateral', 'Default_Retrosplenial',
    'Visual_Lateral', 'Visual_Dorsal_VentralStream', 'Visual_V5', 'Visual_V1',
    'Frontoparietal', 'DorsalAttention', 'Premotor_DorsalAttentionII',
    'Language', 'Salience', 'CinguloOpercular_Action_mode',
    'MedialParietal', 'Somatomotor_Hand', 'Somatomotor_Face',
    'Somatomotor_Foot', 'Auditory', 'SomatoCognitiveAction'
]

# Extract selected features and impute missing values with column means
X = df[columns_to_fit].copy()
X.fillna(X.mean(), inplace=True)

# ========== Standardization + KMeans clustering ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform KMeans clustering (k=3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ========== PCA dimensionality reduction for visualization ==========
# Reduce to 2D space using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Scatter plot: individuals projected on the first two PCA components, colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
plt.title('Clustering of ASD Individuals Based on Structural Network Deviations')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== Step 4: Cluster-wise mean deviation profiles + heatmap ==========
# Compute the average deviation of each network within each cluster
cluster_mean_profiles = df.groupby('Cluster')[columns_to_fit].mean()

print("Mean structural deviations for each cluster:")
print(cluster_mean_profiles)

# Heatmap of average deviations across the 20 brain networks
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_mean_profiles, cmap="coolwarm", center=0, annot=True)
plt.title("Mean Deviation per Cluster Across 20 Structural Networks")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# Convert cluster labels into dummy variables using one-hot encoding
encoder = OneHotEncoder(drop='first', sparse=False)
cluster_onehot = encoder.fit_transform(df[['Cluster']])

# Print first 5 rows as an example
print("One-hot encoded cluster labels (first 5 rows):")
print(cluster_onehot[:5, :])