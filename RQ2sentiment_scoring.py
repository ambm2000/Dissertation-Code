import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Setup
# -------------------------------
TEXT_FOLDER = "TextFiles" 
INPUT_CSV = "output/harm_tagged_policies.csv"
OUTPUT_CSV = "output/harm_sentiment_scored.csv"
os.makedirs("output", exist_ok=True)

# -------------------------------
# Load the harm-tagged data
# -------------------------------
df = pd.read_csv(INPUT_CSV)

# -------------------------------
# Reconstruct file paths and read full text
# -------------------------------
def get_text(row):
    try:
        file_path = os.path.join(TEXT_FOLDER, row['platform_type'], row['platform_name'], row['filename'])
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {row['filename']}: {e}")
        return ""

print("Reading policy texts...")
df['text'] = df.apply(get_text, axis=1)

# -------------------------------
# Apply VADER sentiment analysis
# -------------------------------
print("Applying VADER sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# -------------------------------
# Save updated file
# -------------------------------
df.drop(columns=["text"], inplace=True)  # remove raw text to keep file small
df.to_csv(OUTPUT_CSV, index=False)
print(f"Sentiment scores saved to: {OUTPUT_CSV}")

# -------------------------------
# Visualisations
# -------------------------------
print("Generating sentiment plots...")

# Histogram of sentiment scores
plt.figure(figsize=(8, 5))
sns.histplot(df['sentiment_score'], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Sentiment Scores (VADER Compound)")
plt.xlabel("Sentiment Score (-1 = Negative, +1 = Positive)")
plt.ylabel("Document Count")
plt.tight_layout()
plt.savefig("output/sentiment_histogram.png")
plt.close()

# Boxplot of sentiment scores by platform type
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x='platform_type', y='sentiment_score', palette="pastel")
plt.title("Sentiment Scores by Platform Type")
plt.ylabel("Compound Sentiment Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/sentiment_boxplot.png")
plt.close()

# Bar chart: average sentiment per platform type
avg_sentiment = df.groupby("platform_type")['sentiment_score'].mean().sort_values()
plt.figure(figsize=(8, 5))
avg_sentiment.plot(kind='bar', color='coral')
plt.title("Average Sentiment Score by Platform Type")
plt.ylabel("Mean Compound Sentiment")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/sentiment_avg_by_platform.png")
plt.close()

print("Visualisations saved to /output/")

# -------------------------------
# Create sentiment summary by PLATFORM TYPE
# -------------------------------
summary_by_type = df.groupby("platform_type")['sentiment_score'].agg(
    doc_count='count',
    mean_sentiment='mean',
    median_sentiment='median',
    min_sentiment='min',
    max_sentiment='max',
    std_dev='std'
).reset_index()

# Save to CSV
summary_by_type.to_csv("output/sentiment_summary_by_type.csv", index=False)
print("Sentiment summary by platform type saved: sentiment_summary_by_type.csv")

# -------------------------------
# Create sentiment summary by PLATFORM NAME
# -------------------------------
summary_by_platform = df.groupby("platform_name")['sentiment_score'].agg(
    doc_count='count',
    mean_sentiment='mean',
    median_sentiment='median',
    min_sentiment='min',
    max_sentiment='max',
    std_dev='std'
).reset_index()

# Save to CSV
summary_by_platform.to_csv("output/sentiment_summary_by_platform.csv", index=False)
print("Sentiment summary by platform name saved: sentiment_summary_by_platform.csv")

