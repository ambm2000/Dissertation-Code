import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
client = OpenAI(api_key="API KEY HERE")


# -------------------------------
# Configuration
# -------------------------------
TEXT_FOLDER = "TextFiles"
INPUT_FILE = "output/harm_sentiment_scored.csv"
OUTPUT_FILE = "output/harm_clustered.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 5 
os.makedirs("output", exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
print(" Loading metadata and preparing harm-tagged text...")
df = pd.read_csv(INPUT_FILE)

# Filter to harm-tagged rows only
df = df[df['harm_tags'].map(lambda x: len(eval(x)) > 0)].copy()

# -------------------------------
# Load Raw Text from File
# -------------------------------
def load_raw_text(row):
    try:
        file_path = os.path.join(TEXT_FOLDER, row['platform_type'], row['platform_name'], row['filename'])
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {row['filename']}: {e}")
        return ""

df['text'] = df.apply(load_raw_text, axis=1)

# -------------------------------
# Generate BERT Embeddings
# -------------------------------
print("Generating BERT embeddings...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

# -------------------------------
# KMeans Clustering
# -------------------------------
print(f"Running KMeans clustering (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# -------------------------------
# UMAP Dimensionality Reduction
# -------------------------------
print("Running UMAP for 2D visualisation...")
umap = UMAP(n_components=2, random_state=42)
embedding_2d = umap.fit_transform(embeddings)
df['umap_x'] = embedding_2d[:, 0]
df['umap_y'] = embedding_2d[:, 1]

# -------------------------------
# Visualise UMAP Clusters
# -------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='umap_x', y='umap_y', hue='cluster', palette='tab10', s=60)
plt.title("UMAP: BERT Embedding Clusters of Harm-Tagged Policies")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output/harm_clusters_umap.png")
plt.close()
print("UMAP cluster plot saved: output/harm_clusters_umap.png")

# -------------------------------
# Bar Chart: Cluster Counts by Platform Type
# -------------------------------
print("\nPlotting cluster counts by platform type...")
cluster_counts = df.groupby("platform_type")['cluster'].value_counts().unstack().fillna(0)

cluster_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title("Cluster Distribution by Platform Type")
plt.ylabel("Number of Documents")
plt.xlabel("Platform Type")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output/cluster_distribution_by_platform_type.png")
plt.close()
print("Bar chart saved: cluster_distribution_by_platform_type.png")

# -------------------------------
# Boxplot: Sentiment Score by Cluster
# -------------------------------
print("Plotting sentiment scores by cluster...")
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x='cluster', y='sentiment_score', palette="Set3")
plt.title("Sentiment Score Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Compound Sentiment Score")
plt.tight_layout()
plt.savefig("output/sentiment_by_cluster.png")
plt.close()
print("Boxplot saved: sentiment_by_cluster.png")

# -------------------------------
# Sample Filenames from Each Cluster
# -------------------------------
print("\nExample filenames per cluster:")
for cluster_id in sorted(df['cluster'].unique()):
    print(f"\nCluster {cluster_id} examples:")
    print(df[df['cluster'] == cluster_id]['filename'].sample(min(3, df[df['cluster'] == cluster_id].shape[0]), random_state=42).tolist())

# -------------------------------
# Cluster Summary Outputs
# -------------------------------

from sklearn.feature_extraction import text

# Add all platform names to custom stopword list
platform_names = [
    'badoo', 'bumble', 'call', 'duty', 'hq', 'discord', 'ea', 'sports', 'fc', 'facebook',
    'messenger', 'fortnite', 'google', 'messages', 'grindr', 'hinge', 'instagram',
    'linkedin', 'meta', 'minecraft', 'pinterest', 'quora', 'reddit', 'snapchat', 'telegram',
    'tiktok', 'tinder', 'whatsapp', 'youtube', 'x', 'https', 'com'
]

custom_stopwords = list(text.ENGLISH_STOP_WORDS.union(platform_names))
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_features=1000)

# === TF-IDF Terms Summary ===
print("\nWriting cleaned TF-IDF terms per cluster to CSV...")
tfidf_summary = []

for cluster_id in sorted(df['cluster'].unique()):
    cluster_texts = df[df['cluster'] == cluster_id]['text'].tolist()
    if len(cluster_texts) < 3:
        tfidf_summary.append({'cluster': cluster_id, 'top_terms': 'Too few documents'})
        continue
    X = vectorizer.fit_transform(cluster_texts)
    tfidf_scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
    sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:10]
    top_terms = ', '.join([term for term, _ in sorted_terms])
    tfidf_summary.append({'cluster': cluster_id, 'top_terms': top_terms})

pd.DataFrame(tfidf_summary).to_csv("output/cluster_top_terms.csv", index=False)
print("Saved: output/cluster_top_terms.csv")

# === Platform Name & Type per Cluster ===
print("Writing platform name/type counts per cluster...")

platform_summary = (
    df.groupby(['cluster', 'platform_type'])['platform_name']
    .value_counts()
    .rename('count')
    .reset_index()
)

platform_summary.to_csv("output/platforms_per_cluster.csv", index=False)
print("Saved: output/platforms_per_cluster.csv")

# === Harm Tags per Cluster ===
print("Writing harm tag proportions per cluster...")

# Convert harm_tags from string to list if needed
if isinstance(df['harm_tags'].iloc[0], str):
    df['harm_tags'] = df['harm_tags'].apply(eval)

harm_tag_summary = (
    df.explode('harm_tags')
      .groupby(['cluster', 'harm_tags'])
      .size()
      .reset_index(name='count')
)

harm_tag_summary.to_csv("output/harm_tags_per_cluster.csv", index=False)
print("Saved: output/harm_tags_per_cluster.csv")

# -------------------------------
# Cluster Summary Table (for Report)
# -------------------------------


cluster_labels = {
    0: "Crisis Support & Suicide",
    1: "Legal & Account Terms",
    2: "Abuse Reporting & Safety",
    3: "Content Moderation Guidelines",
    4: "Community Safety & Consent"
}

summary_rows = []

for cluster_id in sorted(df['cluster'].unique()):
    # Average sentiment
    avg_sentiment = df[df['cluster'] == cluster_id]['sentiment_score'].mean()
    
    # Top harm tags
    harm_counts = (
        df[df['cluster'] == cluster_id]
        .explode('harm_tags')['harm_tags']
        .value_counts()
        .head(3)
        .to_dict()
    )
    top_harms = ', '.join([f"{k} ({v})" for k, v in harm_counts.items()])

    # Top platforms
    top_platforms = (
        df[df['cluster'] == cluster_id]['platform_name']
        .value_counts()
        .head(3)
        .to_dict()
    )
    top_platforms_str = ', '.join([f"{k} ({v})" for k, v in top_platforms.items()])

    summary_rows.append({
        'cluster': cluster_id,
        'label': cluster_labels.get(cluster_id, "Undefined"),
        'avg_sentiment': round(avg_sentiment, 3),
        'top_harms': top_harms,
        'top_platforms': top_platforms_str
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("output/cluster_summary_table.csv", index=False)
print("Saved: output/cluster_summary_table.csv")


# -------------------------------
# LLM Summary Table (for Report)
# -------------------------------

def summarise_with_gpt(text, cluster_id):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are a critical language analyst and policy assistant. Your role is to help interpret online platform policies related to safety, abuse, and harm. When given a policy text, identify the document’s main themes, emotional tone, and the types of harm it addresses (e.g., psychological, physical, reputational, sexual, etc.). Be concise but analytical, and prioritise clarity for a researcher studying platform governance."},
                {"role": "user", "content": f"""This is a policy document from the cluster labeled: "{cluster_labels.get(cluster_id, f'Cluster {cluster_id}')}". 

Please provide:

1. A brief 4 sentence summary of what the policy is about.
2. A description of the tone (e.g., supportive, authoritative, punitive).
3. The types of harm addressed (choose from: psychological, physical, reputational, sexual, identity-based, economic, privacy).
4. Any assumptions the platform appears to make about its users or their behaviour.

Here is the document text:

{text}
"""
}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# -------------------------------
# Summarise 10 Representative Docs per Cluster using GPT
# -------------------------------
summary_results = []

for cluster_id in sorted(df['cluster'].unique()):
    print(f"\nSummarising Cluster {cluster_id}...")
    samples = df[df['cluster'] == cluster_id].sample(10, random_state=42)
    for i, row in samples.iterrows():
        sample_text = row['text'][:1500] 
        summary = summarise_with_gpt(sample_text, cluster_id)
        summary_results.append({
            "clusterID": cluster_id,
            "cluster": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
            "platform": row['platform_name'],
            "harm_tags": row['harm_tags'],
            "summary": summary
        })

pd.DataFrame(summary_results).to_csv("output/gpt_cluster_summaries.csv", index=False)
print("Saved: output/gpt_cluster_summaries.csv")

# -------------------------------
# Generate Cluster-Level Meta Summaries using GPT
# -------------------------------

def summarise_cluster_level(cluster_label, doc_summaries):
    try:
        combined_text = "\n\n".join([f"- {s}" for s in doc_summaries])
        prompt = f"""
You are reviewing 10 individual summaries of policy documents that belong to the cluster labeled: "{cluster_label}". These documents were grouped together based on semantic similarity using BERT-based embeddings and clustering.

Your task is to synthesise these summaries and provide a concise and critical analysis that captures the overarching theme, tone, and harm-related focus of the cluster.

Please provide:

1. A 4 sentence meta-summary that captures the main theme or policy focus of the documents in this cluster. Be specific about the subject matter and policy intent.
2. A description of the overall tone used across the documents (e.g., supportive, authoritative, preventative, punitive). Explain how this tone is reflected in the content.
3. The primary types of harm addressed (choose from: psychological, physical, reputational, sexual, identity-based, economic, privacy). List only those that are clearly represented in the summaries.
4. Any patterns or assumptions the platform appears to make about users (e.g., their behaviour, needs, risks, or responsibilities) based on the policy language across the documents.

Here are the 10 summaries from this cluster:

{combined_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert language analyst assisting with academic research on online platform safety policies. Your task is to critically analyse clusters of policy documents that have been automatically grouped based on semantic similarity. You will be given multiple summaries of documents within the same cluster. Identify the overall theme, emotional tone, primary harms addressed, and any implicit assumptions the platform appears to make about its users. Your analysis should be concise, analytical, and written in a clear academic style, suitable for inclusion in a university-level dissertation on platform governance and online harm."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# Load document-level summaries from earlier step
doc_df = pd.read_csv("output/gpt_cluster_summaries.csv")

# Create cluster-level summaries
meta_summary_results = []

for cluster_label in doc_df['cluster'].unique():
    cluster_summaries = doc_df[doc_df['cluster'] == cluster_label]['summary'].tolist()
    meta_summary = summarise_cluster_level(cluster_label, cluster_summaries)
    meta_summary_results.append({
        "cluster_label": cluster_label,
        "meta_summary": meta_summary
    })

pd.DataFrame(meta_summary_results).to_csv("output/gpt_meta_summaries.csv", index=False)
print("Saved: output/gpt_meta_summaries.csv")



# -------------------------------
# Save Output
# -------------------------------
df.drop(columns=["text"], inplace=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Clustered dataset saved to: {OUTPUT_FILE}")
