import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import json

# Define directories
input_dir = "bertopic_results_unsupervised"
output_dir = "bertopic_results_unsupervised_visual"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data with correct column references
topic_distribution = pd.read_csv(os.path.join(input_dir, "topic_distributions.csv"))
topic_summary = pd.read_csv(os.path.join(input_dir, "topic_summary.csv"))
platform_distribution = pd.read_csv(os.path.join(input_dir, "topic_distribution_by_platform.csv"))
topic_category_distribution = pd.read_csv(os.path.join(input_dir, "topic_category_distribution.csv"))
top_terms = pd.read_json(os.path.join(input_dir, "top_terms_per_topic.json"))
topic_similarity = pd.read_csv(os.path.join(input_dir, "topic_coherence_report.csv"))

# Map topic numbers to human-readable names
topic_names = {
    "-1": "General Community Reports & Content",
    "0": "Service Terms & User Rights",
    "1": "Reporting & Help Requests",
    "2": "User Safety & Community Policies",
    "3": "Content Guidelines & Enforcement",
    "4": "Gaming & Player Accounts",
    "5": "Suicide & Self-Harm Policies",
    "6": "Post & Account Violations",
    "7": "Sexual Content & Consent",
    "8": "Hate Speech & Protected Groups",
    "9": "Chat & Messaging Policies",
    "10": "Bullying & Student Protection",
    "11": "Policy Violations & Content Removal",
    "12": "Parental Controls & Teen Safety",
    "13": "Harassment & Targeted Abuse",
    "14": "Spam & Bot Messaging",
    "15": "Blocking & Messenger Controls",
    "16": "Account Bans & Restrictions",
    "17": "Animal Cruelty & Violent Content",
    "18": "Misinformation & Public Health",
    "19": "Extremism & Organisational Violence",
    "20": "Law Enforcement & User Data Requests",
    "21": "Comment Filtering & Moderation"
}

# Apply human-readable topic names
platform_distribution['topic'] = platform_distribution['topic'].astype(str).map(topic_names)
topic_distribution['topic'] = topic_distribution['topic'].astype(str).map(topic_names)
topic_summary['Topic'] = topic_summary['Topic'].astype(str).map(topic_names)
topic_category_distribution['topic'] = topic_category_distribution['topic'].astype(str).map(topic_names)

# === 1. Stacked Bar Chart: Topic Distribution Across Platforms ===
plt.figure(figsize=(12, 6))
platform_pivot = platform_distribution.pivot(index="topic", columns="platform", values="document_count").fillna(0)
platform_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
plt.title("Topic Distribution Across Platforms")
plt.xlabel("Topics")
plt.ylabel("Number of Documents")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Platform", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stacked_bar_topic_distribution.png"))
plt.close()

# === 2. Grouped Bar Chart: Documents per Topic per Platform ===
# Normalize document count as percentage per topic
platform_distribution["normalized_count"] = platform_distribution.groupby("topic")["document_count"].transform(lambda x: x / x.sum() * 100)
plt.figure(figsize=(12, 6))
sns.barplot(data=platform_distribution, x="topic", y="document_count", hue="platform", palette="coolwarm")
plt.figure(figsize=(12, 6))
sns.barplot(data=platform_distribution, x="topic", y="normalized_count", hue="platform", palette="coolwarm")
plt.title("Percentage of Documents Assigned to Each Topic per Platform")
plt.xlabel("Topics")
plt.ylabel("Percentage of Topic Occurrences (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Platform", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 50) 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grouped_bar_topic_distribution_normalized.png"))
plt.close()


# === 3. Bar Chart: Topic Prevalence (Most Frequent Topics) ===
plt.figure(figsize=(12, 6))
topic_summary_sorted = topic_summary.sort_values(by="Count", ascending=False)

sns.barplot(data=topic_summary_sorted, x="Topic", y="Count", hue="Topic", palette="Blues_r", dodge=False, legend=False)

plt.title("Most Frequent Topics in Harassment-Related Policies")
plt.xlabel("Topics")
plt.ylabel("Number of Documents")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_chart_topic_prevalence.png"))
plt.close()

# === 4. Bar Chart: Top Words per Topic (Single Graph) ===

# Load JSON properly as a dictionary (Fixes previous issues)
with open(os.path.join(input_dir, "top_terms_per_topic.json"), "r") as f:
    top_terms = json.load(f)  # Ensures dictionary format

# Convert integer keys to strings (Fixes key mismatch issues)
top_terms = {str(k): v for k, v in top_terms.items()}

# Debugging: Check if JSON keys are correctly formatted
print("Updated Top Terms JSON Keys:", list(top_terms.keys())[:5])  # Should print: ['0', '1', '2', '3', '4']
print("Sample Data for Topic 0:", top_terms.get("0", "No data"))  # Should return a list of words & scores

# Extract top words per topic
top_words_list = []
for topic_id, words_with_scores in top_terms.items():
    if isinstance(words_with_scores, list) and len(words_with_scores) > 0:
        # Extract words and ignore scores
        words_only = [pair[0] for pair in words_with_scores[:5] if isinstance(pair, list) and len(pair) > 1]
        scores = [pair[1] for pair in words_with_scores[:5] if isinstance(pair, list) and len(pair) > 1]
        
        # Debugging: Print extracted words per topic
        print(f"Topic {topic_id}: Words Extracted - {words_only}")

        if words_only:
            topic_name = topic_names.get(topic_id, f"Topic {topic_id}")  # Ensure proper mapping
            for word, score in zip(words_only, scores):
                top_words_list.append((topic_name, word, score))

# Convert extracted words to DataFrame
top_words_df = pd.DataFrame(top_words_list, columns=["Topic", "Word", "Score"])

# Debugging: Check if DataFrame contains data
print("Top Words DataFrame:")
print(top_words_df.head())

# Keep only the top 3 highest-score words per topic
top_words_df = (
    top_words_df.groupby("Topic", group_keys=False)
    .apply(lambda x: x.nlargest(3, "Score"))  # Select top 3 words per topic
    .reset_index(drop=True)  # Reset index to clean up
)


# Ensure DataFrame is not empty before plotting
if not top_words_df.empty:
    plt.figure(figsize=(15, 8))  # Increase size for better readability
    sns.barplot(data=top_words_df, x="Word", y="Score", hue="Topic", palette="magma")

    plt.title("Top Words for Each Topic")
    plt.xlabel("Words")
    plt.ylabel("Importance Score")

    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_chart_top_words_filtered.png"))  # Save with a new name
    plt.close()
else:
    print("⚠ Warning: No data found for Top Words per Topic after filtering.")

# === 5. Word Clouds for Each Topic ===
for topic_id, words in top_terms.items():
    if isinstance(words, list): 
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(map(str, words)))
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud: {topic_names.get(str(topic_id), topic_id)}")
        plt.savefig(os.path.join(output_dir, f"wordcloud_{topic_id}.png"))
        plt.close()

# === 6. Heatmap: Topic Intensity Across Categories ===
heatmap_data = topic_category_distribution.set_index("topic").fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Topic Intensity Across Platform Categories")
plt.xlabel("Platform Categories")
plt.ylabel("Topics")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_topic_intensity.png"))
plt.close()
