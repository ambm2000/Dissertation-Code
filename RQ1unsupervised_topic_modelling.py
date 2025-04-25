import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import json
import torch
import re
from openai import OpenAI

# Ensure Hugging Face parallelism warning is silenced
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU on Apple Silicon to avoid MPS memory errors
if torch.backends.mps.is_available():
    print(" Detected MPS (Apple Silicon GPU), but forcing CPU to avoid memory errors.")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
CLEANED_DIR = "cleaned_texts"
RESULTS_DIR = "bertopic_results_unsupervised"
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of platform names to remove from text before BERTopic
PLATFORM_NAMES = [
    "call of duty hq", "ea sports fc 24", "fortnite", "minecraft",
    "facebook messenger", "google messages", "snapchat", "telegram", "whatsapp", "discord",
    "facebook", "instagram", "linkedin", "meta", "pinterest", "reddit", "tiktok", "x", "youtube",
    "badoo", "bumble", "grindr", "hinge", "tinder", "epic", "duty", "call", "microsoft", "snap", "ea", "google"
]


def preprocess_text(text):
    """Removes platform names from the text before topic modelling."""
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in PLATFORM_NAMES) + r')\b'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)  # Remove platform names completely


def collect_documents():
    """ Collects and pre-processes documents, while keeping metadata for later topic analysis. """
    documents = []
    metadata = []
    doc_count = 0

    print("Starting document collection...")

    for category in os.listdir(CLEANED_DIR):
        category_path = os.path.join(CLEANED_DIR, category)
        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")

        for platform in os.listdir(category_path):
            platform_path = os.path.join(category_path, platform)
            if not os.path.isdir(platform_path):
                continue

            print(f"   Processing platform: {platform}")

            for filename in os.listdir(platform_path):
                file_path = os.path.join(platform_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()
                    cleaned_text = preprocess_text(raw_text)  # Remove platform names
                    documents.append(cleaned_text)
                    metadata.append({
                        "category": category,
                        "platform": platform,
                        "filename": filename
                    })
                    doc_count += 1
                    if doc_count % 50 == 0:
                        print(f"      Processed {doc_count} documents so far...")

    print(f"Document collection complete. Total documents processed: {doc_count}")
    assert doc_count == 530, f"Expected 530 documents, but found {doc_count}."

    return documents, pd.DataFrame(metadata)

def train_bertopic(documents):
    """ Trains an unsupervised BERTopic model to extract topics from pre-processed documents. """
    print("\nStarting BERTopic training...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=3)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=5
    )

    topics, probs = topic_model.fit_transform(documents)

    print(f"BERTopic training complete. Total topics found: {len(topic_model.get_topics())}")
    return topic_model, topics, probs

def save_topic_info(topic_model):
    """ Saves topic summary. Platform names were already removed before training. """
    print("Saving topic summary...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(RESULTS_DIR, "topic_summary.csv"), index=False)
    print(f"Topic summary saved to {RESULTS_DIR}/topic_summary.csv")

def save_document_topic_mapping(df_meta, topics, probs):
    """ Saves document-topic mapping while keeping platform metadata for later comparisons. """
    df_meta["topic"] = topics
    df_meta["probability"] = probs
    df_meta.to_csv(os.path.join(RESULTS_DIR, "topic_distributions.csv"), index=False)
    print(f"Document-topic mapping saved to {RESULTS_DIR}/topic_distributions.csv")

def save_topic_distribution_by_platform(df_meta):
    """ Saves topic distributions per platform. """
    print("Calculating topic distributions across individual platforms...")

    platform_distribution = (
        df_meta.groupby(["platform", "topic"])
        .size()
        .reset_index(name="document_count")
    )

    platform_distribution.to_csv(os.path.join(RESULTS_DIR, "topic_distribution_by_platform.csv"), index=False)
    print(f"Platform-wise topic distribution saved to {RESULTS_DIR}/topic_distribution_by_platform.csv")

def save_topic_distribution_by_category(df_meta):
    """ Saves topic distributions per category. """
    print("Calculating topic distributions across platform categories...")

    category_distribution = (
        df_meta.groupby(["category", "topic"])
        .size()
        .reset_index(name="document_count")
    )

    category_distribution_pivot = category_distribution.pivot(
        index="topic", columns="category", values="document_count"
    ).fillna(0)

    category_distribution_pivot.to_csv(os.path.join(RESULTS_DIR, "topic_category_distribution.csv"))
    print(f"Category-wise topic distribution saved to {RESULTS_DIR}/topic_category_distribution.csv")

def save_top_terms(topic_model):
    """ Saves top terms per topic for manual review. """
    print("Saving top terms per topic for manual review...")
    top_terms_per_topic = {
        topic_id: topic_model.get_topic(topic_id)
        for topic_id in range(len(topic_model.get_topics()))
    }
    with open(os.path.join(RESULTS_DIR, "top_terms_per_topic.json"), "w", encoding="utf-8") as f:
        json.dump(top_terms_per_topic, f, indent=4)
    print(f"Top terms per topic saved to {RESULTS_DIR}/top_terms_per_topic.json")

def save_topic_coherence_report(topic_model):
    """ Generates a structured report for manual topic analysis. """
    print("Generating topic coherence report...")

    topic_info = topic_model.get_topic_info()

    coherence_report = []
    for topic_id in topic_info["Topic"].unique():
        if topic_id == -1:
            continue  # Skip outliers
        top_words = topic_model.get_topic(topic_id)[:10]
        top_words = ", ".join([word for word, _ in top_words])
        coherence_report.append({"Topic": topic_id, "Top Words": top_words})

    coherence_df = pd.DataFrame(coherence_report)
    coherence_df.to_csv(os.path.join(RESULTS_DIR, "topic_coherence_report.csv"), index=False)

    print(f"Topic coherence report saved to {RESULTS_DIR}/topic_coherence_report.csv")

def main():
    documents, df_meta = collect_documents()

    topic_model, topics, probs = train_bertopic(documents)

    save_topic_info(topic_model)
    save_document_topic_mapping(df_meta, topics, probs)
    save_topic_distribution_by_platform(df_meta)
    save_topic_distribution_by_category(df_meta)
    save_top_terms(topic_model)
    save_topic_coherence_report(topic_model)

    print("\nAll outputs saved. BERTopic processing complete.")

if __name__ == "__main__":
    main()
