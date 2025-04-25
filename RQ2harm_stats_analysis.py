import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
from scipy.stats import chi2_contingency
import os

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("output/harm_tagged_policies.csv")

# -------------------------------
# Descriptive Statistics
# -------------------------------
print("\nDescriptive Statistics")
total_docs = len(df)
docs_with_harms = df[df['harm_tags'].map(lambda x: len(eval(x)) > 0)]
print(f"Total documents: {total_docs}")
print(f"Documents with harm tags: {len(docs_with_harms)} ({round(len(docs_with_harms)/total_docs*100, 2)}%)")

# -------------------------------
# Harm frequency overall
# -------------------------------

# Define mapping from harm column names to readable labels
harm_label_map = {
    "harm_psychological": "Psychological",
    "harm_reputational": "Reputational",
    "harm_physical": "Physical",
    "harm_sexual": "Sexual",
    "harm_identity_based": "Identity based",
    "harm_privacy": "Privacy",
    "harm_economic": "Economic",
}

harm_columns = list(harm_label_map.keys())
harm_counts = docs_with_harms[harm_columns].sum().rename(harm_label_map).sort_values(ascending=False)

# Bar plot for harm frequencies
plt.figure(figsize=(10, 5))
harm_counts.plot(kind='bar', color='steelblue')
plt.title("Frequency of Harm Types in Tagged Policies")
plt.ylabel("Number of Documents")
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/harm_type_frequency.png")
plt.close()
print("Bar chart saved: harm_type_frequency.png")

# -------------------------------
# Harm frequency by platform type
# -------------------------------
print("\nComparing Across Platform Types")
platform_harm_counts = docs_with_harms.groupby("platform_type")[harm_columns].sum()
platform_harm_counts = platform_harm_counts.rename(columns=harm_label_map)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(platform_harm_counts)
print(f"Chi-square test: χ² = {chi2:.2f}, p = {p:.4f}")

# Stacked bar chart
platform_harm_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title("Harm Type Distribution by Platform Type")
plt.ylabel("Number of Documents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/harm_type_by_platform.png")
plt.close()
print("Stacked bar chart saved: harm_type_by_platform.png")

# -------------------------------
# Harm Co-occurrence Matrix
# -------------------------------
print("\nHarm Co-occurrence Analysis")

# Helper to extract pairs
def extract_pairs(tag_str):
    tags = eval(tag_str)
    return list(combinations(sorted(set(tags)), 2)) if len(tags) > 1 else []

# Count co-occurrences
cooccurrence_pairs = docs_with_harms['harm_tags'].map(extract_pairs)
all_pairs = [pair for sublist in cooccurrence_pairs for pair in sublist]
pair_counts = Counter(all_pairs)

# Build matrix
harm_types = [col.replace("harm_", "") for col in harm_columns]
co_matrix = pd.DataFrame(0, index=harm_types, columns=harm_types)

for (a, b), count in pair_counts.items():
    co_matrix.loc[a, b] = count
    co_matrix.loc[b, a] = count

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(co_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Harm Type Co-occurrence Heatmap")
plt.tight_layout()
plt.savefig("output/harm_cooccurrence_heatmap.png")
plt.close()
print("Heatmap saved: harm_cooccurrence_heatmap.png")

# -------------------------------
# Save stats to CSV
# -------------------------------
harm_counts.to_csv("output/harm_type_frequency.csv", header=["count"])
platform_harm_counts.to_csv("output/harm_by_platform_type.csv")
co_matrix.to_csv("output/harm_cooccurrence_matrix.csv")
print("Summary data saved to /output/")

# -------------------------------
# Harm frequency heatmap by individual platform (empty cells instead of 0)
# -------------------------------
print("\nGenerating heatmap for harm types by platform (empty cells for 0s)...")

# Define harm label map
harm_label_map = {
    "harm_psychological": "Psychological",
    "harm_reputational": "Reputational",
    "harm_physical": "Physical",
    "harm_sexual": "Sexual",
    "harm_identity_based": "Identity based",
    "harm_privacy": "Privacy",
    "harm_economic": "Economic",
}

harm_columns = list(harm_label_map.keys())
df['platform_name'] = df['platform_name'].astype(str)

# Group by platform and sum harm columns
platform_harm_matrix = df.groupby("platform_name")[harm_columns].sum()

# Rename columns for readability
platform_harm_matrix = platform_harm_matrix.rename(columns=harm_label_map)

# Drop platforms with no harm counts at all
platform_harm_matrix = platform_harm_matrix[platform_harm_matrix.sum(axis=1) > 0]

# Create a string version for annotations (empty string for 0, string for >0)
annot_data = platform_harm_matrix.copy()
annot_data = annot_data.applymap(lambda x: str(int(x)) if x > 0 else '')

# Plot heatmap
plt.figure(figsize=(12, max(6, 0.4 * len(platform_harm_matrix))))
sns.heatmap(platform_harm_matrix, annot=annot_data, fmt='', cmap='YlOrRd', linewidths=0.5, cbar=True)
plt.title("Harm Type Frequency by Platform")
plt.xlabel("Harm Type")
plt.ylabel("Platform")
plt.tight_layout()
plt.savefig("output/harm_heatmap_by_platform.png")
plt.close()

print("Heatmap saved: output/harm_heatmap_by_platform.png")


