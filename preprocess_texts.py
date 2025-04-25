import os
import json
import re
from collections import Counter
import spacy
from nltk.corpus import stopwords
import nltk

# Ensure nltk stopwords are downloaded
nltk.download('stopwords')

# Load English stopwords and spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Define directories
BASE_DIR = os.getcwd()
TEXT_DIR = os.path.join(BASE_DIR, "TextFiles")
OUTPUT_DIR = os.path.join(BASE_DIR, "cleaned_texts")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to clean and normalise text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to tokenize, remove stopwords, and lemmatise
def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stop_words
    ]
    return tokens

# Function to process all files and calculate stats
def process_all_files(expected_total_files=530):
    category_stats = {}
    total_processed_files = 0

    print("\nStarting text preprocessing...")

    # Walk through categories (e.g., SocialMedia, Dating, Messaging, Gaming)
    for category in os.listdir(TEXT_DIR):
        category_path = os.path.join(TEXT_DIR, category)

        if not os.path.isdir(category_path):
            continue

        print(f"\nProcessing category: {category}")

        total_words = 0
        total_tokens = 0
        unique_tokens = set()
        token_counter = Counter()

        # Walk through platform folders inside each category
        for platform in os.listdir(category_path):
            platform_path = os.path.join(category_path, platform)

            if not os.path.isdir(platform_path):
                continue

            print(f"   Processing platform: {platform}")

            # Output path preserving category and platform structure
            platform_output_dir = os.path.join(OUTPUT_DIR, category, platform)
            os.makedirs(platform_output_dir, exist_ok=True)

            for filename in os.listdir(platform_path):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(platform_path, filename)
                output_path = os.path.join(platform_output_dir, filename)

                print(f"      Processing file ({total_processed_files + 1}/{expected_total_files}): {filename}")

                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                cleaned_text = clean_text(raw_text)
                tokens = preprocess_text(cleaned_text)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(tokens))

                print(f"      Completed cleaning: {filename}")

                total_words += len(cleaned_text.split())
                total_tokens += len(tokens)
                unique_tokens.update(tokens)
                token_counter.update(tokens)

                total_processed_files += 1

        category_stats[category] = {
            "total_files": total_processed_files,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "unique_tokens": len(unique_tokens),
            "avg_tokens_per_file": total_tokens / total_processed_files if total_processed_files else 0,
            "top_10_tokens": [token for token, _ in token_counter.most_common(10)],
        }

        print(f"Finished processing category: {category}")
        print(f"   Files processed: {total_processed_files}")
        print(f"   Total words: {total_words}, Total tokens: {total_tokens}")
        print(f"   Top 10 tokens: {category_stats[category]['top_10_tokens']}")

    # Save category stats to JSON
    summary_path = os.path.join(BASE_DIR, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(category_stats, f, indent=4)

    # Final check - did we process all expected files?
    print(f"\nFinal check: Processed {total_processed_files} files (Expected: {expected_total_files})")

    if total_processed_files == expected_total_files:
        print("\nAll files successfully processed!")
    else:
        print(f"\nWARNING: Expected {expected_total_files} files, but only processed {total_processed_files} files.")
        print("Please check for missing files or issues in the TextFiles directory.")

    print(f"Summary saved to: {summary_path}")

# Main script
if __name__ == "__main__":
    process_all_files(expected_total_files=530)
