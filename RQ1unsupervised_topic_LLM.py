import os
import pandas as pd
from openai import OpenAI

# Configuration
client = OpenAI(api_key="API KEY HERE")
TEXT_DIR = "TextFiles"
RESULTS_DIR = "bertopic_results_unsupervised"

# === Load raw document text ===
def load_text_file(row):
    try:
        path = os.path.join(TEXT_DIR, row['category'], row['platform'], row['filename'])
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR loading text: {str(e)}"

# === GPT document-level summarisation ===
def summarise_topic_with_gpt(text, topic_id):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a critical language analyst supporting academic research on online safety and platform governance. Your role is to interpret online policy documents and provide concise, analytical summaries. For each text, identify its core purpose, emotional tone, and the types of harm it seeks to address (e.g., psychological, physical, reputational, sexual, identity-based, economic, privacy). Ensure your summaries are objective, clear, and written in a formal academic style suitable for inclusion in a dissertation or research report."},
                {"role": "user", "content": f"""You are reviewing a policy document that belongs to topic {topic_id}, identified through BERTopic modelling of 530 online platform safety policies. This topic groups together documents with semantically similar approaches to safety, governance, or harm mitigation.

Your task is to carefully analyse this single document and provide a structured, academic-style response with the following components:

1. **Summary**: A 4-sentence summary capturing the main objective and regulatory focus of the policy.
2. **Tone**: Describe the emotional or rhetorical tone used (e.g., supportive, authoritative, preventative, punitive), and briefly justify your answer.
3. **Types of Harm Addressed**: Identify the specific types of harm the policy focuses on, choosing from: psychological, physical, reputational, sexual, identity-based, economic, privacy. Include only those clearly evident.
4. **User Assumptions**: What assumptions (explicit or implicit) does the platform appear to make about its users, their behaviour, or their responsibilities?

Ensure your answer is clear, analytical, and well-structured.

Document text:
{text}
"""}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# === GPT topic name generation ===
def generate_topic_name(top_words, topic_id):
    try:
        prompt = f"""You are helping name a topic from BERTopic modelling of online safety policies.

Here are the top representative terms for Topic {topic_id}:
{top_words}

Based on these terms, provide a short, human-readable label for this topic (e.g., “Self-Harm & Crisis Response”, “Data Sharing Policies”). The name should reflect the common theme or purpose implied by the terms and be appropriate for academic writing."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert academic researcher specialising in natural language processing and topic modelling. Your task is to assign clear, concise, and human-readable labels to automatically generated topics from BERTopic. Each topic consists of a ranked list of its most representative terms. Your goal is to distil the essence of the topic into a short, descriptive phrase (3–6 words) that accurately reflects the core theme or subject matter of the terms. Prioritise clarity, academic tone, and usefulness for inclusion in a research paper analysing online safety policies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Topic {topic_id}"
    
# === GPT topic-level meta-summary ===
def generate_topic_meta_summary(topic_id, topic_label, summaries):
    try:
        combined_text = "\n\n".join([f"- {s}" for s in summaries])
        prompt = f"""
You are reviewing 10 individual summaries of online platform policy documents, all of which belong to Topic {topic_id}: "{topic_label}". These documents were grouped together using BERTopic, based on shared semantic and thematic content related to online safety and governance.

Your task is to synthesise these summaries into a formal, academic-style meta-analysis that reflects the collective intent and characteristics of the topic. Focus on clarity, analytical depth, and policy relevance, using language suitable for a university-level dissertation.

Please provide:

1. A 4-sentence summary that clearly explains the overarching policy focus of this topic, including any common regulatory goals, safety interventions, or behavioural guidance.
2. An assessment of the overall tone conveyed across the documents (e.g., supportive, authoritative, preventative, punitive). Briefly justify this tone using cues from the summaries.
3. The primary types of harm addressed within this topic (select from: psychological, physical, reputational, sexual, identity-based, economic, privacy). Include only those clearly evident across multiple summaries.
4. Any implicit or explicit assumptions that platforms appear to make about user behaviour, needs, risks, or responsibilities based on the content and structure of the policies.

Here are the 10 summaries to base your synthesis on:

{combined_text}
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior academic researcher and expert in language analysis, specialising in the interpretation of BERTopic-generated topics from online safety policy documents. Your role is to produce concise, formal, and analytically rigorous summaries that translate abstract topic representations into clear academic insights. Each summary must capture the underlying regulatory focus, rhetorical tone, types of harm addressed, and any implicit assumptions about users or platform behaviour. Your writing should be precise, structured, and suitable for inclusion in a university-level dissertation on platform governance and digital safety."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# === Main summary pipeline ===
def run_gpt_topic_summaries():
    df_meta = pd.read_csv(os.path.join(RESULTS_DIR, "topic_distributions.csv"))
    topic_info = pd.read_csv(os.path.join(RESULTS_DIR, "topic_summary.csv"))
    
    summaries = []
    topic_name_map = {}

    for topic_id in sorted(df_meta['topic'].unique()):
        if topic_id == -1:
            continue  # Skip outliers

        top_words_row = topic_info[topic_info["Topic"] == topic_id]
        top_words = top_words_row["Representation"].values[0] if not top_words_row.empty else "unknown"
        topic_label = generate_topic_name(top_words, topic_id)
        topic_name_map[topic_id] = topic_label

        print(f"\nSummarising Topic {topic_id}: {topic_label}")
        topic_subset = df_meta[df_meta['topic'] == topic_id]
        n_docs = min(10, len(topic_subset))
        topic_rows = topic_subset.sample(n_docs, random_state=42)


        for _, row in topic_rows.iterrows():
            raw_text = load_text_file(row)
            short_text = raw_text[:1500]
            summary = summarise_topic_with_gpt(short_text, topic_id)
            summaries.append({
                "topic_id": topic_id,
                "topic_name": topic_label,
                "platform": row['platform'],
                "filename": row['filename'],
                "summary": summary
            })

    pd.DataFrame(summaries).to_csv(os.path.join(RESULTS_DIR, "gpt_topic_summaries.csv"), index=False)
    print("\nSaved: gpt_topic_summaries.csv")

    print("\nGenerating topic-level meta-summaries...")
    df_summaries = pd.DataFrame(summaries)
    meta_rows = []

    for topic_id, group in df_summaries.groupby("topic_id"):
        topic_label = group["topic_name"].iloc[0]
        doc_summaries = group["summary"].tolist()
        meta_summary = generate_topic_meta_summary(topic_id, topic_label, doc_summaries)
        meta_rows.append({
            "topic_id": topic_id,
            "topic_name": topic_label,
            "meta_summary": meta_summary
        })

    pd.DataFrame(meta_rows).to_csv(os.path.join(RESULTS_DIR, "gpt_topic_meta_summaries.csv"), index=False)
    print("Saved: gpt_topic_meta_summaries.csv")


if __name__ == "__main__":
    run_gpt_topic_summaries()
