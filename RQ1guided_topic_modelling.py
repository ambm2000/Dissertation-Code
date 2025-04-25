import os
import pandas as pd
import json
import torch
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import re

# === CONFIGURATION ===
CLEANED_DIR = "cleaned_texts"  # Input directory
RESULTS_DIR_BASE = "bertopic_results"  # Base output directory for different approaches
os.makedirs(RESULTS_DIR_BASE, exist_ok=True)  # Ensure base directory exists


# === DEVICE CONFIGURATION ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === PREDEFINED TOPICS & KEYWORDS ===
predefined_topics = {
    "Private & Direct Message Harassment": [
        "message", "unsolicited", "spam", "privacy", "harassment", "unwanted",
        "contact", "inbox", "abuse", "threat", "manipulation", "coercion"
    ],
    "Voice & Audio-Based Harassment": [
        "voice", "toxicity", "abuse", "slur", "verbal",
        "chat", "threat", "insult", "mocking", "ridicule", "profanity", "exclusion"
    ],
    "Competitive & Fair Play Harassment": [
        "cheating", "hacking", "toxicity", "match",
        "unfair", "exclusion", "exploit", "misconduct", "sabotage", 
        "griefing", "gatekeeping", "smurfing"
    ],
    "Gender-Based Harassment": [
        "sexism", "misogyny", "discrimination", "hate", "LGBTQ", "feminist",
        "gender", "slur", "abuse", "intimidation", "demeaning", "bias"
    ],
    "Content Moderation & Platform Enforcement": [
        "violation", "removal", "strike", "appeal", "suspension", "restriction",
        "ban evasion", "account penalty", "AI detection", "community rules", "trust & safety"
    ],
    "Self-Harm & Mental Health-Related Harassment": [
        "suicide", "selfharm", "disorder", "eat", "wellness",
        "prevention", "intervention", "crisis", "harmful", "triggering", "distress", "vulnerability"
    ],
    "Sexual Harassment & Coercion": [
        "harassment", "unwanted", "explicit", "sexting", "coercion",
        "predator", "consent", "abuse", "inappropriate", "grooming",
        "objectification", "catfishing"
    ],
    "Revenge Porn & Non-Consensual Intimate Content": [
        "revenge", "porn", "deepfake", "leak", "explicit", "blackmail",
        "sextortion", "nude", "unauthorized", "exploitation", "humiliation", "privacy invasion"
    ],
    "Hate Speech & Public Discrimination": [
        "racism", "extremist", "slur", "xenophobia", "political",
        "incitement", "hate", "propaganda", "supremacist", "dehumanization",
        "dehumanisation", "radicalization", "radicalisation", "speech incitement", "dogwhistling"
    ],
    "Targeted Discriminatory Harassment": [
        "abuse", "racial", "gender", "LGBTQ", "discrimination", "threat",
        "targeted", "microaggression", "harassment", "exclusion", "intimidation", "hostility"
    ],
    "Cyberbullying, Doxxing & Stalking": [
        "bullying", "doxxing", "stalking", "revenge", "cancel",
        "location", "exposure", "intimidation", "tracking", "impersonation",
        "blackmail", "harassment campaign"
    ],
    "Cross-Platform Harassment & Ban Evasion": [
        "account", "sockpuppet", "evade", "migration", "username",
        "loophole", "repeat", "coordinated", "persistent", "shadowing",
        "alternative accounts", "ban circumvention", "brigading"
    ],
    "Violent & Harmful Content in Harassment": [
        "violence", "threat", "extremist", "terrorism", "gore",
        "incitement", "radicalisation", "glorification", "assault",
        "brutality", "mob harassment", "incel ideology"
    ],
    "Misinformation & Harassment-Related Disinformation": [
        "fake", "news", "misinformation", "conspiracy", "deepfake",
        "propaganda", "deception", "manipulation", "hoax", "misleading",
        "narrative control", "fabricated evidence"
    ],
    "Law Enforcement & Legal Compliance": [
        "subpoena", "digital", "takedown", "cybercrime", "compliance",
        "court", "liability", "regulation", "legal", "legislation",
        "investigation", "prosecution"
    ],
    "AI Moderation, Algorithmic Bias & False Positives": [
        "AI", "algorithm", "bias", "automation", "false",
        "detection", "oversight", "transparency", "system failure",
        "unfair", "AI-driven removal", "machine learning bias", "hallucinations"
    ],
    "Immersive & Spatial Harassment": [
        "VR", "metaverse", "spatial", "avatar", "cyber",
        "presence", "groping", "whispering", "surveillance",
        "intrusion", "immersive", "online embodiment"
    ]
}

PLATFORM_NAMES = [
    "call of duty hq", "ea sports fc 24", "fortnite", "minecraft",
    "facebook messenger", "google messages", "snapchat", "telegram", "whatsapp", "discord",
    "facebook", "instagram", "linkedin", "meta", "pinterest", "reddit", "tiktok", "x", "youtube",
    "badoo", "bumble", "grindr", "hinge", "tinder", "epic", "duty", "call", "microsoft", "snap", "ea"
]

# Convert topics into BERTopic format
seed_topic_list = list(predefined_topics.values())

# === LOGGING FUNCTION ===
def log(message):
    print(f"[LOG] {message}")

# === TEXT PREPROCESSING FUNCTION ===
def clean_text(text, platform_names, remove_platforms=True, replace_with_placeholder=False):
    """
    Cleans text by either removing platform names or replacing them with "PLATFORM_NAME".

    :param text: Input text.
    :param platform_names: List of platform names.
    :param remove_platforms: If True, removes platform names.
    :param replace_with_placeholder: If True, replaces platform names with "PLATFORM_NAME".
    :return: Processed text.
    """
    pattern = r'\b(?:' + '|'.join(re.escape(name) for name in platform_names) + r')\b'

    if remove_platforms:
        return re.sub(pattern, '', text, flags=re.IGNORECASE) 

    if replace_with_placeholder:
        return re.sub(pattern, 'PLATFORM_NAME', text, flags=re.IGNORECASE) 

    return text  

# === DOCUMENT COLLECTION FUNCTION ===
def collect_documents(remove_platforms=True, replace_with_placeholder=False):
    """
    Reads files, processes text, and stores metadata with platform handling.

    :param remove_platforms: If True, removes platform names.
    :param replace_with_placeholder: If True, replaces platform names with "PLATFORM_NAME".
    :return: Processed documents & metadata DataFrame.
    """
    documents = []
    metadata = []
    doc_count = 0

    log("Starting document collection...")

    for category in os.listdir(CLEANED_DIR):
        category_path = os.path.join(CLEANED_DIR, category)
        if not os.path.isdir(category_path):
            continue

        log(f"Processing category: {category}")

        for platform in os.listdir(category_path):
            platform_path = os.path.join(category_path, platform)
            if not os.path.isdir(platform_path):
                continue

            log(f"   Processing platform: {platform}")

            for filename in os.listdir(platform_path):
                file_path = os.path.join(platform_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()

                    # Clean text based on selected method
                    cleaned_text = clean_text(raw_text, PLATFORM_NAMES, remove_platforms, replace_with_placeholder)

                    documents.append(cleaned_text)
                    metadata.append({
                        "category": category,
                        "platform": platform,
                        "filename": filename,
                        "original_text": raw_text  # Store original text for comparison
                    })

                    doc_count += 1
                    if doc_count % 50 == 0:
                        log(f"      Processed {doc_count} documents...")

    log(f"Document collection complete. Total documents processed: {doc_count}")
    return documents, pd.DataFrame(metadata)

# === BERTopic TRAINING FUNCTION ===
def train_guided_bertopic(documents):
    """
    Trains BERTopic model using predefined topics (guided mode).
    """
    log(f"Starting Guided BERTopic training...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=3)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        seed_topic_list=seed_topic_list,
        min_topic_size=5
    )

    topics, probs = topic_model.fit_transform(documents)

    log(f"Guided BERTopic training complete. Topics found: {len(topic_model.get_topics())}")
    return topic_model, topics, probs

# === SAVE RESULTS FUNCTION ===
def save_results(topic_model, topics, df_meta):
    """
    Saves guided BERTopic results into 'bertopic_results_guided/'.
    - Replaces topic numbers with topic names.
    - Removes 'original_text' from topic_distributions.csv.
    - Saves topic-word associations in JSON and CSV format.
    """
    results_dir = "bertopic_results_guided"
    log(f"Saving guided BERTopic results in: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)  # Ensure directory exists

    # Ensure 'guided_topic' exists before mapping names
    df_meta["guided_topic"] = topics  # Assign topics to metadata DataFrame

    # Retrieve topic names and replace topic numbers with names
    topic_labels = topic_model.get_topic_info()[["Topic", "Name"]].set_index("Topic").to_dict()["Name"]
    df_meta["guided_topic"] = df_meta["guided_topic"].map(topic_labels)

    # Save topic summary
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(results_dir, "topic_summary.csv"), index=False)

    # Remove 'original_text' column before saving topic distributions
    df_meta.drop(columns=["original_text"], inplace=True, errors="ignore")
    df_meta.to_csv(os.path.join(results_dir, "topic_distributions.csv"), index=False)

    # Save topic distributions by platform (with topic names)
    platform_distribution = df_meta.groupby(["platform", "guided_topic"]).size().reset_index(name="document_count")
    platform_distribution.to_csv(os.path.join(results_dir, "topic_distribution_by_platform.csv"), index=False)

    # Save topic distributions by category (with topic names)
    category_distribution = df_meta.groupby(["category", "guided_topic"]).size().reset_index(name="document_count")
    category_distribution_pivot = category_distribution.pivot(index="guided_topic", columns="category", values="document_count").fillna(0)
    category_distribution_pivot.to_csv(os.path.join(results_dir, "topic_category_distribution.csv"))

    # Extract topic words and their scores
    top_terms = topic_model.get_topics()

    # Save top words per topic in JSON format
    top_terms_cleaned = {
        topic_id: [word for word, _ in words]  # Extract only words, ignore scores
        for topic_id, words in top_terms.items() if words is not None
    }
    with open(os.path.join(results_dir, "top_terms_per_topic.json"), "w", encoding="utf-8") as f:
        json.dump(top_terms_cleaned, f, indent=4)

    # Save words and their scores in CSV format
    topic_word_scores = []
    for topic_id, words in top_terms.items():
        if words is not None:
            for word, score in words:
                topic_word_scores.append([topic_labels.get(topic_id, f"Topic {topic_id}"), word, score])

    topic_word_scores_df = pd.DataFrame(topic_word_scores, columns=["Topic Name", "Word", "Score"])
    topic_word_scores_df.to_csv(os.path.join(results_dir, "topic_word_scores.csv"), index=False)

    log(f"Guided BERTopic results saved in {results_dir}.")

# === MAIN FUNCTION ===
def main():
    log("Starting Guided BERTopic Pipeline...")

    log(f"Running BERTopic with 'Removed' approach...")

    # Load documents with platform names removed
    documents, df_meta = collect_documents(remove_platforms=True, replace_with_placeholder=False)

    # Train BERTopic
    topic_model_guided, topics_guided, probs_guided = train_guided_bertopic(documents)

    # Save results to 'bertopic_results_guided'
    save_results(topic_model_guided, topics_guided, df_meta)

    log("Guided BERTopic Pipeline Completed!")

# === RUN SCRIPT ===
if __name__ == "__main__":
    main()
