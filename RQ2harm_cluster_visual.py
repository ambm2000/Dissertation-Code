import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv("output/harm_clustered.csv")

# -------------------------------
# Cluster label mapping
# -------------------------------
cluster_labels = {
    0: "Crisis Support & Suicide",
    1: "Legal & Account Terms",
    2: "Abuse Reporting & Safety",
    3: "Content Moderation Guidelines",
    4: "Community Safety & Consent"
}

df['cluster_label'] = df['cluster'].map(cluster_labels)

# -------------------------------
# Compute heatmap data
# -------------------------------
heatmap_data = pd.crosstab(df['platform_name'], df['cluster_label'])

# Drop platforms with 0 total policies
heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1) > 0]

# -------------------------------
# Plot heatmap
# -------------------------------
plt.figure(figsize=(12, max(6, 0.4 * len(heatmap_data))))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="YlGnBu", linewidths=0.5, cbar=True)
plt.title("Distribution of Harm Clusters by Platform")
plt.xlabel("Harm Cluster")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Platform")
plt.tight_layout()

# -------------------------------
# Save output
# -------------------------------
output_folder = "output/Used"
os.makedirs(output_folder, exist_ok=True)
plt.savefig(f"{output_folder}/harm_clusters_by_platform.png")
plt.close()

print("Heatmap saved to output/Used/harm_clusters_by_each_platform.png")
