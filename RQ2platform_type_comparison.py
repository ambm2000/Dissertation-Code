import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

os.makedirs("output", exist_ok=True)

# Load the input file
df = pd.read_csv("output/harm_clustered.csv")

# -------------------------------
# Step 1: Harm Distribution by Platform Type
# -------------------------------
df["harm_tags"] = df["harm_tags"].apply(eval) 
df_exploded = df.explode("harm_tags")
harm_dist = df_exploded.groupby(["platform_type", "harm_tags"]).size().unstack().fillna(0)

# Save harm distribution
harm_dist.to_csv("output/step5_harm_distribution_by_platform.csv")

# Plot harm distribution
plt.figure(figsize=(12, 6))
harm_dist.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 6))
plt.title("Harm Type Distribution by Platform Type")
plt.ylabel("Number of Documents")
plt.xlabel("Platform Type")
plt.tight_layout()
plt.savefig("output/step5_harm_distribution_by_platform.png")
plt.close()

# -------------------------------
# Step 2: Chi-Squared Test
# -------------------------------
chi2, p, dof, expected = chi2_contingency(harm_dist)
with open("output/step5_chi_squared_result.txt", "w") as f:
    f.write(f"Chi-squared test statistic: {chi2:.4f}\n")
    f.write(f"Degrees of freedom: {dof}\n")
    f.write(f"P-value: {p:.6f}\n")

# -------------------------------
# Step 3: Sentiment by Platform Type
# -------------------------------
sentiment_avg = df.groupby("platform_type")["sentiment_score"].mean()
sentiment_avg.to_csv("output/step5_avg_sentiment_by_platform.csv")

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_avg.index, y=sentiment_avg.values, palette="coolwarm")
plt.title("Average Sentiment Score by Platform Type")
plt.ylabel("Average Sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/step5_avg_sentiment_by_platform.png")
plt.close()

# -------------------------------
# Step 4: Cluster Distribution by Platform Type
# -------------------------------

# Define cluster label mapping
cluster_label_map = {
    0: "0 – Crisis Support & Suicide",
    1: "1 – Legal & Account Terms",
    2: "2 – Abuse Reporting & Safety",
    3: "3 – Content Moderation Guidelines",
    4: "4 – Community Safety & Consent"
}

# Map cluster numbers to labels
df["cluster_label"] = df["cluster"].map(cluster_label_map)

# Group by platform type and cluster label
cluster_dist = df.groupby(["platform_type", "cluster_label"]).size().unstack().fillna(0).astype(int)
cluster_dist.to_csv("output/step5_cluster_distribution_by_platform.csv")

ordered_labels = [cluster_label_map[i] for i in sorted(cluster_label_map.keys())]

cluster_dist[ordered_labels].plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Accent")
plt.title("Cluster Distribution by Platform Type")
plt.ylabel("Number of Documents")
plt.xlabel("Platform Type")
plt.tight_layout()
plt.savefig("output/step5_cluster_distribution_by_platform.png")
plt.close()
