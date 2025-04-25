import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# === CONFIGURATION ===
client = OpenAI(api_key="API KEY HERE")
GUIDED_IN = "bertopic_results_guided/gpt_guided_meta_summaries.csv"
UNSUP_IN = "bertopic_results_unsupervised/gpt_topic_meta_summaries.csv"
GUIDED_OUT = "bertopic_results_guided/gpt_guided_meta_summaries_tagged.csv"
UNSUP_OUT = "bertopic_results_unsupervised/gpt_topic_meta_summaries_tagged.csv"

TONE_CHOICES = ["supportive", "authoritative", "preventative", "procedural", "directive", "neutral-formal", "cautious"]
HARM_CHOICES = ["psychological", "identity-based", "privacy-related", "reputational", "physical", "economic", "sexual", "public health", "content-related", "other"]

# === CLASSIFICATION FUNCTION ===
def classify_summary(summary_text):
    prompt = f"""
You are analysing a summary of a policy topic. Extract two sets of labels:

1. **Tone(s)** — choose one or more from: {", ".join(TONE_CHOICES)}  
2. **Harm Types** — choose one or more from: {", ".join(HARM_CHOICES)}

Only return the labels, in the following format:
Tone: [comma-separated list]
Harm Types: [comma-separated list]

Summary:
{summary_text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a digital safety policy analyst. Extract tone and harm types from summaries for academic tagging."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()

        tone_line = next((line for line in content.splitlines() if line.lower().startswith("tone:")), "")
        harm_line = next((line for line in content.splitlines() if line.lower().startswith("harm types:")), "")
        tones = [t.strip() for t in tone_line.split(":", 1)[1].split(",")] if ":" in tone_line else []
        harms = [h.strip() for h in harm_line.split(":", 1)[1].split(",")] if ":" in harm_line else []

        return tones, harms
    except Exception as e:
        print(f"Error tagging summary: {e}")
        return [], []

# === PROCESSING FUNCTION ===
def tag_file(input_path, output_path):
    df = pd.read_csv(input_path)
    tones, harms = [], []

    print(f"\nProcessing {input_path}")
    for summary in tqdm(df["meta_summary"], desc="Classifying"):
        tone_tags, harm_tags = classify_summary(summary)
        tones.append(", ".join(tone_tags))
        harms.append(", ".join(harm_tags))

    df['tone'] = tones
    df['harm_types'] = harms
    df.drop(columns=['meta_summary'], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return df

# === STATS PRINTER ===
def print_stats(df, label):
    print(f"\n=== {label} Statistics ===")
    tone_counts = df['tone'].str.split(', ').explode().value_counts()
    harm_counts = df['harm_types'].str.split(', ').explode().value_counts()
    multi_harm = df['harm_types'].str.contains(',').sum()
    tone_harm_pairs = df.assign(
        tone=df['tone'].str.split(', '),
        harm_types=df['harm_types'].str.split(', ')
    ).explode('tone').explode('harm_types').groupby(['tone', 'harm_types']).size()

    print("\nTone distribution:")
    print(tone_counts.to_string())
    print("\nHarm type distribution:")
    print(harm_counts.to_string())
    print(f"\nTopics with multiple harm types: {multi_harm}")
    print("\nTone-Harm pairs:")
    print(tone_harm_pairs.to_string())

# === MAIN PIPELINE ===
def main():
    df_guided = tag_file(GUIDED_IN, GUIDED_OUT)
    df_unsup = tag_file(UNSUP_IN, UNSUP_OUT)

    print_stats(df_guided, "Guided Topics")
    print_stats(df_unsup, "Unsupervised Topics")

if __name__ == "__main__":
    main()
