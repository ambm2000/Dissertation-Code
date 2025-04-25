import os
import pandas as pd
import re

# -----------------------------------------------
# Define Harm Typology Dictionary
# -----------------------------------------------
harm_dict = {
    "psychological": [
        "trauma", "anxiety", "emotional distress", "fear", "depression", "mental abuse",
        "psychological harm", "panic", "emotional injury", "disturbing content", "self-esteem harm", "stress"
    ],
    "reputational": [
        "defamation", "libel", "false information", "character attack", "reputation damage",
        "public shaming", "false accusations", "name-calling", "rumors", "inaccurate claims", "malicious statements"
    ],
    "physical": [
        "death threats", "violence", "stalking", "physical harm", "injury", "kill", "assault",
        "bodily threat", "danger", "threats to safety", "intimidation", "harassment leading to violence"
    ],
    "sexual": [
        "sexual harassment", "revenge porn", "non-consensual imagery", "cyber-flashing", "nudity",
        "sexual threat", "sexual content", "sexually explicit", "sexual violence", "sexual abuse",
        "inappropriate photos", "sexual solicitation"
    ],
    "identity_based": [
        "hate speech", "racism", "misogyny", "homophobia", "transphobia", "Islamophobia",
        "antisemitism", "slur", "xenophobia", "harmful stereotypes", "ethnic abuse", "religious insult"
    ],
    "privacy": [
        "doxxing", "leak", "unauthorized sharing", "IP address", "location exposure", "personal data breach",
        "non-consensual exposure", "private info", "data misuse", "privacy violation", "tracking", "surveillance"
    ],
    "economic": [
        "job loss", "professional damage", "income threat", "career harm", "employment retaliation",
        "scam", "fraud", "economic abuse", "financial loss", "malicious complaints", "boycotts", "blackmail"
    ]
}

# -----------------------------------------------
# Harm Tagging Function
# -----------------------------------------------
def tag_harms(text, harm_dict):
    tags = set()
    text_lower = text.lower()
    for category, keywords in harm_dict.items():
        for word in keywords:
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                tags.add(category)
                break  # Stop checking further keywords in this category
    return list(tags)

# -----------------------------------------------
# Traverse TextFiles Folder and Process Files
# -----------------------------------------------
data = []
total_files = 0
files_with_harms = 0

base_folder = "TextFiles"
print(f" Starting harm tagging from base folder: {base_folder}\n")

for platform_type in os.listdir(base_folder):
    type_path = os.path.join(base_folder, platform_type)
    if not os.path.isdir(type_path):
        continue

    print(f"Processing platform type: {platform_type}")

    for platform_name in os.listdir(type_path):
        platform_path = os.path.join(type_path, platform_name)
        if not os.path.isdir(platform_path):
            continue

        print(f"   Platform: {platform_name}")

        for filename in os.listdir(platform_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(platform_path, filename)
                total_files += 1

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    harm_tags = tag_harms(text, harm_dict)
                    if harm_tags:
                        files_with_harms += 1

                    row = {
                        "filename": filename,
                        "platform_name": platform_name,
                        "platform_type": platform_type,
                        "harm_tags": harm_tags
                    }

                    for category in harm_dict.keys():
                        row[f"harm_{category}"] = category in harm_tags

                    data.append(row)

                    print(f"      Processed: {filename} — Tags: {harm_tags if harm_tags else 'None'}")

                except Exception as e:
                    print(f"      Error reading {filename}: {e}")

# -----------------------------------------------
# Save Output
# -----------------------------------------------
df = pd.DataFrame(data)
output_path = "output/harm_tagged_policies.csv"
os.makedirs("output", exist_ok=True)
df.to_csv(output_path, index=False)

print("\nSummary:")
print(f"Total files processed: {total_files}")
print(f"Files with harm tags: {files_with_harms}")
print(f"Output saved to: {output_path}")
