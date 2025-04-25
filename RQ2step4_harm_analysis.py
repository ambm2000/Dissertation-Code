import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations

# === CONFIG ===
INPUT_FILE = "output/harm_clustered.csv"  
OUTPUT_DIR = "output/step4_cooccurrence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Load Data ===
df = pd.read_csv(INPUT_FILE)

if isinstance(df['harm_tags'].iloc[0], str):
    df['harm_tags'] = df['harm_tags'].apply(eval)

# === 2. Extract Harm Pairs ===
def extract_pairs(harm_list):
    if len(harm_list) < 2:
        return []
    return list(combinations(sorted(set(harm_list)), 2))

df['harm_pairs'] = df['harm_tags'].apply(extract_pairs)

# === 3. Count Frequencies ===
all_pairs = [pair for pairs in df['harm_pairs'] for pair in pairs]
pair_counts = Counter(all_pairs)

# === 4. Create Co-occurrence Matrix ===
harm_types = sorted({harm for tags in df['harm_tags'] for harm in tags})
co_matrix = pd.DataFrame(0, index=harm_types, columns=harm_types)

for (a, b), count in pair_counts.items():
    co_matrix.loc[a, b] += count
    co_matrix.loc[b, a] += count  # symmetric

# Save raw matrix
co_matrix.to_csv(os.path.join(OUTPUT_DIR, "harm_cooccurrence_matrix.csv"))

# === 5. Visualise Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(co_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Harm Type Co-occurrence Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "harm_cooccurrence_heatmap.png"))
plt.close()
print("Co-occurrence matrix and heatmap saved.")

# === 6. Breakdown by Cluster or Platform Type ===
def generate_group_heatmap(group_col):
    for group_val in df[group_col].unique():
        subset = df[df[group_col] == group_val].copy()
        group_pairs = [pair for pairs in subset['harm_pairs'] for pair in pairs]
        group_counts = Counter(group_pairs)

        group_matrix = pd.DataFrame(0, index=harm_types, columns=harm_types)
        for (a, b), count in group_counts.items():
            group_matrix.loc[a, b] += count
            group_matrix.loc[b, a] += count

        # Save and plot
        group_matrix.to_csv(os.path.join(OUTPUT_DIR, f"cooccurrence_{group_col}_{group_val}.csv"))

        plt.figure(figsize=(10, 8))
        sns.heatmap(group_matrix, annot=True, fmt="d", cmap="YlOrBr", linewidths=0.5)
        plt.title(f"Harm Co-occurrence: {group_col} = {group_val}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{group_col}_{group_val}.png"))
        plt.close()
        print(f"  Saved heatmap for {group_col}: {group_val}")

print("\nGenerating breakdowns by cluster and platform type...")
generate_group_heatmap("cluster")
generate_group_heatmap("platform_type")

print("\nStep 4 complete. All outputs saved to:", OUTPUT_DIR)
