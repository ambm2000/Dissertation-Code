import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from wordcloud import WordCloud

# Directories
INPUT_DIR = "bertopic_results_guided"
OUTPUT_DIR = "bertopic_results_guided_visual"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topic Name Mapping
TOPIC_NAME_MAPPING = {
    "-1_content_report_community_account": "Community Guidelines & Reports",
    "0_safety_community_information_user": "Safety & Privacy",
    "1_chat_message_messenger_block": "Messaging & Chat Moderation",
    "2_game_player_voice_account": "Gaming & Voice Chat Safety",
    "3_suicide_disorder_eat_selfharm": "Self-Harm & Mental Health",
    "4_service_term_use_agreement": "Terms of Service & Agreements",
    "5_sexual_consent_explicit_content": "Sexual Content & Consent",
    "6_post_account_report_violation": "Account Reports & Violations",
    "7_hate_protect_speech_group": "Hate Speech & Protected Groups",
    "8_bully_bullying_student_help": "Bullying & Student Support",
    "9_report_click_message_account": "Message Reports & Actions",
    "10_information_datum_device_use": "User Data & Information",
    "11_harassment_bullying_individual_target": "Targeted Harassment & Bullying",
    "12_teen_parent_comment_young": "Teen & Parental Safety",
    "13_spam_server_report_account": "Spam & Fake Accounts",
    "14_account_ban_appeal_error": "Bans & Appeal Errors",
    "15_strike_restriction_policy_violation": "Policy Violations & Restrictions",
    "16_appeal_pin_deactivate_request": "Account Appeals & Deactivations",
    "17_trans_shame_body_report": "Body Shaming & Gender Harassment",
    "18_animal_violence_violent_harm": "Animal & Physical Violence",
    "19_misinformation_false_content_health": "Health Misinformation",
    "20_content_edsa_guideline_misinformation": "Misinformation & Guidelines",
    "21_violence_violent_organisation_extremist": "Violent Extremism & Terrorism",
    "22_request_law_enforcement_user": "Legal Requests & Law Enforcement",
    "23_job_report_select_match": "User Matches & Reporting",
    "24_filter_comment_word_message": "Comment & DM Filtering",
    "25_member_boundary_ghost_feel": "Community Boundaries & Ghosting"
}

### Topic Distribution by Platform (Bar Chart & Stacked Bar Chart) ###
df_platform = pd.read_csv(os.path.join(INPUT_DIR, "topic_distribution_by_platform.csv"))

# Replace topic names with human-readable labels
df_platform["guided_topic"] = df_platform["guided_topic"].map(TOPIC_NAME_MAPPING)

# Pivot table for stacked bar chart
df_pivot = df_platform.pivot(index="guided_topic", columns="platform", values="document_count").fillna(0)

# Bar chart
plt.figure(figsize=(12, 6))
df_platform.groupby("guided_topic")["document_count"].sum().sort_values().plot(kind="barh", color="skyblue")
plt.xlabel("Document Count")
plt.ylabel("Guided Topic")
plt.title("Topic Distribution by Platform")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "topic_distribution_by_platform_bar.png"))
plt.close()

# Stacked bar chart
df_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab10")
plt.xlabel("Guided Topic")
plt.ylabel("Document Count")
plt.title("Stacked Topic Distribution by Platform")
plt.legend(title="Platform", bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "topic_distribution_by_platform_stacked.png"))
plt.close()


### Topic Distribution by Category (Heatmap) ###
df_category = pd.read_csv(os.path.join(INPUT_DIR, "topic_category_distribution.csv"))

# Replace topic names with human-readable labels
df_category["guided_topic"] = df_category["guided_topic"].map(TOPIC_NAME_MAPPING)
df_category.set_index("guided_topic", inplace=True)
df_category.fillna(0, inplace=True)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_category, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5)
plt.xlabel("Platform Category")
plt.ylabel("Guided Topic")
plt.title("Topic Distribution by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "topic_distribution_by_category_heatmap.png"))
plt.close()


### Top Words Per Topic (Word Clouds) ###
with open(os.path.join(INPUT_DIR, "top_terms_per_topic.json"), "r", encoding="utf-8") as f:
    top_terms = json.load(f)

for topic_id, words in top_terms.items():
    topic_name = TOPIC_NAME_MAPPING.get(topic_id, f"Topic {topic_id}")  # Default if not in mapping

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {topic_name}")
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"wordcloud_{topic_name.replace(' ', '_')}.png"))
    plt.close()


### Topic Representation Strength (Boxplot & Violin Plot) ###
df_word_scores = pd.read_csv(os.path.join(INPUT_DIR, "topic_word_scores.csv"))

# Replace topic names with human-readable labels
df_word_scores["Topic Name"] = df_word_scores["Topic Name"].map(TOPIC_NAME_MAPPING)

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_word_scores, x="Topic Name", y="Score", palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Guided Topic")
plt.ylabel("Word Score")
plt.title("Topic Representation Strength (Boxplot)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "topic_representation_boxplot.png"))
plt.close()

# Violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_word_scores, x="Topic Name", y="Score", palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Guided Topic")
plt.ylabel("Word Score")
plt.title("Topic Representation Strength (Violin Plot)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "topic_representation_violin.png"))
plt.close()

print("Visualisations saved in:", OUTPUT_DIR)
