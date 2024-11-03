# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage  # For dendrogram

# Load the dataset
data = pd.read_csv("data.csv")
print("Dataset loaded successfully!\n")
print(data.head())

# Select relevant columns
features = data[['danceability', 'energy', 'loudness', 'tempo']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Function to plot clusters with professional color palette
def plot_clusters(data, labels, title, palette="Blues_d"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette, s=60, marker='o', edgecolor='w', linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend([],[], frameon=False)  # Remove legend for a cleaner look
    plt.show()

# Initial Clustering: Display with professional color palette
plot_clusters(scaled_features, np.random.randint(0, 5, len(scaled_features)), 'Initial Random Clusters', palette="crest")

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Plot K-Means Clusters with a professional color scheme
plot_clusters(scaled_features, kmeans_labels, 'K-Means Clustering', palette="mako")

# Generate and plot Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
linkage_matrix = linkage(scaled_features, method='ward')  # 'ward' is commonly used for dendrograms
dendrogram(linkage_matrix)
plt.title('Dendrogram for Agglomerative Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Final clusters and error metrics for both clustering methods
print("\nK-Means Cluster Labels:", kmeans_labels)

# Error Metrics
print("\n--- Error and Evaluation Metrics ---")
error_rate_kmeans = kmeans.inertia_
print("K-Means Error Rate (Inertia):", error_rate_kmeans)

silhouette_kmeans = silhouette_score(scaled_features, kmeans_labels)
silhouette_agglo = silhouette_score(scaled_features, AgglomerativeClustering(n_clusters=5).fit_predict(scaled_features))

print("K-Means Silhouette Score:", silhouette_kmeans)
print("Agglomerative Silhouette Score:", silhouette_agglo)
