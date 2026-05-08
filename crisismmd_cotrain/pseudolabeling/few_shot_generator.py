import pandas as pd
import json
import random
import os

# def set_seed(seed: int):
#     """
#     Sets random, numpy, torch, and torch.cuda seeds
#     """

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# ---------------- CONFIG ---------------- #
input_file = "data/llm_pseudo_labels/informative/Llama-3.2-11B-Instruct-Info-Zeroshot-Text-Only.json"
output_dir = "data/llm_pseudo_labels/informative/few_shot_splits/Llama-3.2-11B-Instruct-Info-Zeroshot-Text-Only-extended"

# shot_sizes = [1, 5, 10, 20, 35] # Humanitarian few-shot 
shot_sizes = [1000, 500, 250, 200, 150, 100, 50, 20, 10, 5, 1] # Informative 

random_seed = 42
random.seed(random_seed)
os.makedirs(output_dir, exist_ok=True)

# ---------------- LOAD DATA ---------------- #
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
if isinstance(data, dict):
    df = pd.DataFrame.from_dict(data, orient='index')
else:
    df = pd.DataFrame(data)

print(f"✅ Loaded {len(df)} samples from {input_file}")

# ---------------- GROUP BY LABEL ---------------- #
label_col = "ori_label"
tweet_id_col = "tweet_id"
grouped = df.groupby(label_col)

#print(grouped.size())


unique_labels = grouped.ngroups
print(f"🧩 Found {unique_labels} unique labels: {list(grouped.groups.keys())}")

# ---------------- SELECT 300 PER LABEL ---------------- #

num_groups = 2000
labeled_dfs = []
for label, group in grouped:
    #print(f"label: {label} group: {group} \n")
    if len(group) >= num_groups:
        sampled = group.sample(num_groups, random_state=random_seed)
    labeled_dfs.append(sampled)

labeled_df = pd.concat(labeled_dfs, ignore_index=True)
#print(f"label len before {len(labeled_df)}")

labeled_ids = set(labeled_df[tweet_id_col])
#print(f"label len after {len(labeled_ids)}")

unlabeled_df = df[~df[tweet_id_col].isin(labeled_ids)]
#print(f"unlabel len before {len(unlabeled_df)}")

#print(f"unlabel len after {len(unlabeled_df.nunique())}")

print(f"🎯 Selected {unique_labels} × {num_groups} = {len(labeled_df)} labeled samples total.")
print(f"   Remaining {len(unlabeled_df)} samples go to unlabeled base.\n")

set1_list, set2_list = [], []

for label, group in labeled_df.groupby(label_col):
    # select 2×shot samples per label
    chosen = group.sample(2 * 1000, random_state=random_seed)

    set1 = chosen.iloc[:1000]
    set2 = chosen.iloc[1000:2000]

    set1_list.append(set1)
    set2_list.append(set2)

# ---------------- FEW-SHOT SPLITS ---------------- #
for shot in shot_sizes:

    

    # combine all labels
    set1_df = pd.concat(set1_list, ignore_index=True)
    set2_df = pd.concat(set2_list, ignore_index=True)

    # prepare unlabeled
    used_ids = set(set1_df[tweet_id_col]).union(set2_df[tweet_id_col])
    remaining_labeled = labeled_df[~labeled_df[tweet_id_col].isin(used_ids)]
    unlabeled_split = pd.concat([remaining_labeled, unlabeled_df], ignore_index=True)

    # ---------------- SAVE ---------------- #
    out1 = os.path.join(output_dir, f"info_{shot}-shot_train_set1.json")
    out2 = os.path.join(output_dir, f"info_{shot}-shot_train_set2.json")
    unlabeled_out = os.path.join(output_dir, f"unlabeled_train_{shot}.json")

    set1_df.to_json(out1, orient='index', force_ascii=False, indent=4)
    set2_df.to_json(out2, orient='index', force_ascii=False, indent=4)
    unlabeled_split.to_json(unlabeled_out, orient='index', force_ascii=False, indent=4)

    print(f"✅ {shot}-shot split complete ({shot}×{unique_labels} per set).")
    print(f"   - {out1}")
    print(f"   - {out2}")
    print(f"   - {unlabeled_out}\n")

print("🎉 All label-balanced few-shot datasets created successfully using Pandas.")
