import csv
import json

# Input TSV file path
input_file = "data/crisismmd_splits/task_humanitarian_text_img_agreed_lab_test.tsv"

# Read TSV file
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

# Remove duplicate tweet_id (keep first occurrence)
unique_texts = {}
for row in rows:
    tid = row["tweet_id"]
    if tid not in unique_texts:
        unique_texts[tid] = row

unique_text_rows = list(unique_texts.values())

unique_images = {}
for row in rows:
    tid = row["image_id"]
    if tid not in unique_images:
        unique_images[tid] = row

unique_image_rows = list(unique_images.values())

# Prepare dictionaries (indexed)
text_only = {}
image_only = {}

# Process each unique row
for i, row in enumerate(unique_text_rows):

    entry = {
        "id": i,
        "tweet_id": row["tweet_id"],
        "event_name": row["event_name"],
        "label": row["label"],
        "tweet_text": row["tweet_text"]
    }
    text_only[i] = entry

# Convert to JSON strings (orient='index')
json_text = json.dumps(text_only, indent=2, ensure_ascii=False)

# Process each unique row
for i, row in enumerate(unique_image_rows):

    entry = {
        "id": i,
        "tweet_id": row["tweet_id"],
        "event_name": row["event_name"],
        "label": row["label"],
        "tweet_text": row["tweet_text"],
        "image_path": row["image"],
        "image_id": row["image_id"]
    }
    image_only[i] = entry

json_image = json.dumps(image_only, indent=2, ensure_ascii=False)

# Assuming you already have:
# json_text, json_image, json_text_image

# Define output file names
text_file = "data/CrisisMMD_Modified/humanitarian/test/text_only.json"
image_file = "data/CrisisMMD_Modified/humanitarian/test/image_only.json"

# Write each JSON string to a file
with open(text_file, "w", encoding="utf-8") as f1:
    f1.write(json_text)

with open(image_file, "w", encoding="utf-8") as f2:
    f2.write(json_image)

print(f"✅ Files saved successfully:\n - {text_file}\n - {image_file}")
