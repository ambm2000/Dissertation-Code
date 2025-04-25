import os
import pandas as pd
from openai import OpenAI

# === CONFIGURATION ===
client = OpenAI(api_key="API KEY NEEDED HERE")
TEXT_DIR = "TextFiles"
RESULTS_DIR = "bertopic_results_guided"

# === Load raw document text ===
def load_text_file(row):
    try:
        path = os.path.join(TEXT_DIR, row['category'], row['platform'], row['filename'])
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR loading text: {str(e)}"

# === GPT Document-Level Summarisation ===
def summarise_topic_with_gpt(text, topic_name):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an academic language analyst contributing to a university-level research project on online safety, harm regulation, and platform governance. Your role is to critically interpret individual safety policy documents and generate formal, structured summaries suitable for inclusion in a dissertation or peer-reviewed publication. For each document, provide an academically styled synthesis that identifies the document’s core regulatory purpose, rhetorical tone, specific types of harm addressed, and any assumptions made about users’ behaviours, vulnerabilities, or responsibilities. Your analysis should be concise, precise, and written in a clear and formal academic register that reflects expert understanding of digital policy discourse."},
                {"role": "user", "content": f"""
You are reviewing a platform safety policy document grouped under the BERTopic-derived topic: "{topic_name}". This topic was identified through semantic clustering of 530 online policy documents focused on digital harm, user safety, and governance frameworks.

Your task is to produce a structured, academically rigorous interpretation of this individual policy document. Focus on regulatory framing, policy objectives, and harm-related language.

Please provide:

1. **Summary**: A concise 4-sentence academic summary describing the document’s main objectives, regulatory scope, and intended user interventions.
2. **Tone**: Describe the dominant rhetorical tone of the policy (e.g., supportive, preventative, punitive, authoritative) and briefly justify your classification based on content or phrasing.
3. **Types of Harm Addressed**: Identify the types of harm explicitly or substantively discussed in the policy. Choose only those clearly evidenced from this typology: psychological, physical, reputational, sexual, identity-based, economic, privacy.
4. **User Assumptions**: Note any inferred or stated assumptions about users—such as their behavioural risks, capacity for self-regulation, vulnerability, or expected responsibilities.

Use concise, formal academic language appropriate for a postgraduate dissertation on digital safety and policy design.

Here is the document excerpt:
{text[:1500]}
"""}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# === GPT Topic-Level Meta-Summary ===
def summarise_topic_meta(topic_name, summaries):
    try:
        combined_text = "\n\n".join([f"- {s}" for s in summaries])
        prompt = f"""
You are reviewing 10 document-level summaries drawn from the BERTopic-guided topic: "{topic_name}". These policy documents were clustered based on semantic similarity and a shared focus on online safety, harm mitigation, or platform governance.

Your task is to synthesise these summaries into a single, academically structured topic-level analysis. This synthesis will support qualitative interpretation of the topic and contribute to the analysis of online platform policy framing.

Please address the following four elements in your output:

1. **Thematic Summary (4 sentences):** Describe the overarching theme and regulatory intent of the topic, highlighting commonalities in how platforms frame or address online safety concerns.
2. **Tone Description:** Characterise the overall tone used across the documents (e.g., supportive, preventative, punitive), and explain how this tone is conveyed linguistically or through policy content.
3. **Types of Harm Addressed:** Identify the dominant types of harm referenced in the documents. Select from the following typology: psychological, physical, reputational, sexual, identity-based, economic, privacy.
4. **Platform Assumptions:** Summarise any common assumptions made by platforms about users, including perceived risks, behavioural expectations, or capacities for self-regulation.

Here are the 10 summaries to base your synthesis on:
{combined_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an academic language analyst and topic modelling researcher. Your task is to produce formal, structured meta-analyses of topics derived from BERTopic (guided or unsupervised) applied to online platform safety policy documents. Each topic represents a cluster of semantically similar texts with a shared regulatory or thematic focus. Your summaries should be written in a concise and analytical style, suitable for inclusion in a postgraduate dissertation on platform governance and online harms. For each topic, you will synthesise a set of document-level summaries into a cohesive interpretation. Focus on identifying the topic’s overarching purpose, dominant tone, types of harm addressed, and implicit assumptions about user behaviour or risk. Write in clear, formal academic prose."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# === Run summarisation pipeline ===
def run_guided_gpt_pipeline():
    df_meta = pd.read_csv(os.path.join(RESULTS_DIR, "topic_distributions.csv"))
    df_summary = pd.read_csv(os.path.join(RESULTS_DIR, "topic_summary.csv"))

    summaries = []
    meta_summaries = []

    topic_names = df_summary.set_index("Topic")["Name"].to_dict()

    for topic_name in sorted(df_meta['guided_topic'].unique()):
        if topic_name == "-1" or pd.isna(topic_name):
            continue

        print(f"\nSummarising topic: {topic_name}")
        topic_docs = df_meta[df_meta['guided_topic'] == topic_name]
        n = min(10, len(topic_docs))
        selected_docs = topic_docs.sample(n, random_state=42)

        doc_summaries = []

        for _, row in selected_docs.iterrows():
            raw_text = load_text_file(row)
            summary = summarise_topic_with_gpt(raw_text, topic_name)
            summaries.append({
                "topic": topic_name,
                "platform": row["platform"],
                "filename": row["filename"],
                "summary": summary
            })
            doc_summaries.append(summary)

        # Generate topic-level meta-summary
        meta = summarise_topic_meta(topic_name, doc_summaries)
        meta_summaries.append({
            "topic": topic_name,
            "meta_summary": meta
        })

    # Save outputs
    pd.DataFrame(summaries).to_csv(os.path.join(RESULTS_DIR, "gpt_guided_doc_summaries.csv"), index=False)
    print("Saved: gpt_guided_doc_summaries.csv")

    pd.DataFrame(meta_summaries).to_csv(os.path.join(RESULTS_DIR, "gpt_guided_meta_summaries.csv"), index=False)
    print("Saved: gpt_guided_meta_summaries.csv")

if __name__ == "__main__":
    run_guided_gpt_pipeline()
