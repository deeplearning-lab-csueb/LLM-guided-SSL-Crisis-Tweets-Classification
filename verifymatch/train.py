import argparse
import csv
import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyparsing as pp
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (  
    get_constant_schedule, 
    get_constant_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

# Added event class mapping
EVENT_CLASS_MAPPING = {
    'california_wildfires_2018': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'missing_or_found_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'canada_wildfires_2016': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'cyclone_idai_2019': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'missing_or_found_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'hurricane_dorian_2019': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'hurricane_florence_2018': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'hurricane_harvey_2017': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'hurricane_irma_2017': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'hurricane_maria_2017': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'kaikoura_earthquake_2016': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support'],
    'kerala_floods_2018': ['caution_and_advice', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'not_humanitarian', 'other_relevant_information', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort', 'sympathy_and_support']
}

from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from torch.distributions.distribution import Distribution
from tqdm import tqdm
from torch.distributions import Categorical
from itertools import cycle

from datasets import load_from_disk, load_dataset
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from run_humaid import get_paths

import wandb

# Pick the largest value the platform allows
SAFE_LIMIT = 2**31 - 1  # max signed 32-bit int
csv.field_size_limit(min(sys.maxsize, SAFE_LIMIT))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs (rec: 12-18)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--warmup_steps',type=int, default=0)
parser.add_argument('--gradient_accumulation_steps',default=1)
parser.add_argument('--labeled_train_path', type=str, help='labeled train dataset path')
parser.add_argument('--unlabeled_train_path', type=str, help='unlabeled train dataset path')
parser.add_argument('--ssl',action='store_true',help='Semi-supervised learning')
parser.add_argument('--mixup',action='store_true')
parser.add_argument('--pseudo_label_by_normalized',type=bool)
parser.add_argument('--same_domain_unlabeled',action='store_true')
parser.add_argument('--th', type=float, default=0.7)
parser.add_argument('--sharpening',action='store_true',default=True)
parser.add_argument('--T',type=float,default=0.1)
parser.add_argument('--seed',type=int,default=int(time.time()))
parser.add_argument('--rand_mixup',action='store_true')
parser.add_argument('--mixup_loss_weight',type=float, default=1.)
parser.add_argument('--consistency',action='store_true')
parser.add_argument('--high_mixup',action='store_true',default=False)
parser.add_argument('--multigpus',action='store_true')
parser.add_argument('--unlabeled_batch_size',type=int,default=32)
parser.add_argument('--set_num', type=int, required=True, help='set number (1, 2, or 3)')
parser.add_argument('--event', type=str, required=True, help='event name')
parser.add_argument('--lbcl', type=str, required=True, help='label count')
parser.add_argument('--keep_local_ckpt', action='store_true',
                    help="Keep local .pt files after logging artifacts (default: delete after upload).")
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
args = parser.parse_args()

def random_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Construct optional grouping and tagging info
args.do_train = True
args.do_evaluate = True
# args.pseudo_label_by_normalized = True
args.ssl = True
args.mixup = True

event = args.event
lbcl = args.lbcl
run_num = os.getenv("WANDB_RUN_ID", str(int(time.time())))

group_name = f'{event}_{lbcl}'
tags = [args.task,event,lbcl]

# Initialize W&B run safely
wandb.init(
    project="humaid_ssl",
    config=vars(args),
    mode="online",
    group=group_name,
    tags=tags
)

config = wandb.config

# Override args with sweep config values
args.learning_rate = config.learning_rate
args.weight_decay = config.weight_decay
args.batch_size = config.batch_size
args.epochs = config.epochs
args.T = config.T
args.mixup_loss_weight = config.mixup_loss_weight
args.label_smoothing = config.label_smoothing
args.max_grad_norm = config.max_grad_norm
args.th = config.th
args.task = config.task
args.model = config.model
args.max_seq_length = config.max_seq_length


args.seed = config.seed
args.task = config.task
args.model = config.model
args.max_seq_length = config.max_seq_length

args.learning_rate = config.learning_rate
args.batch_size = config.batch_size
# args.batch_size = 8
args.T = config.T
args.epochs = config.epochs
# args.epochs = 1

# unique run name (timestamped) to avoid overwriting
wandb.run.name = (
    f"{args.task}-{event}"
    f"-lb{lbcl}-{int(time.time())}"
)

print(args)

assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG', 'SICK','RTE','FEVER','HANS','CrisisMMDINF', 'HumAID')
assert args.model in ('bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'vinai/bertweet-base')
if args.task in ('HumAID'):
    if args.event in EVENT_CLASS_MAPPING:
        n_classes = len(EVENT_CLASS_MAPPING[args.event])
        print(f"Using {n_classes} classes for event: {args.event}")
    else:
        n_classes = 10
        print(f"Warning: Event {args.event} not found in mapping. Using default 10 classes.")
elif args.task in ('SNLI', 'MNLI','SICK','FEVER','HANS'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB','RTE','CrisisMMDINF'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1

def cuda(tensor):
    """Places tensor on CUDA device."""
    if args.multigpus:
        return tensor.cuda()
    else:
        return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_single_inputs(sentence):
    """
    Encodes a single-sentence input for pre-trained models using the template
    [CLS] sentence [SEP].
    Returns input_ids, segment_ids, and attention_mask — same format and dtype
    as encode_pair_inputs().
    """
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )

    input_ids = inputs['input_ids']

    # Use segment IDs only if BERT-type model (same logic as pair encoding)
    if args.model in ('bert-base-uncased', 'bert-large-uncased'):
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0] * len(input_ids)

    attention_mask = [1] * len(input_ids)
    padding_length = args.max_seq_length - len(input_ids)

    # Apply same padding logic
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    # Sanity check (same as pair version)
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length

    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0]*len(inputs['input_ids'])
    attention_mask = [1]*len(inputs['input_ids'])#inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_mc_inputs(context, start_ending, endings):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        inputs = tokenizer.encode_plus(
            context, start_ending+" " + ending, add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
            segment_ids = inputs['token_type_ids']
        else:
            segment_ids = [0] * len(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_masks.append(attention_mask)
    return (
        cuda(torch.tensor(all_input_ids)).long(),
        cuda(torch.tensor(all_segment_ids)).long(),
        cuda(torch.tensor(all_attention_masks)).long(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()

class HumAIDProcessor:
    """Data loader for HumAID."""

    def __init__(self, event=None):
        if event and event in EVENT_CLASS_MAPPING:
            # Dynamically create label map based on alphabetized class list
            classes = EVENT_CLASS_MAPPING[event]
            self.label_map = {label: i for i, label in enumerate(classes)}
        else:
            # Fallback to default
            self.label_map = {'requests_or_urgent_needs': 0, 
                            'rescue_volunteering_or_donation_effort': 1, 'infrastructure_and_utility_damage': 2, 
                            'missing_or_found_people': 3, 'displaced_people_and_evacuations': 4, 
                            'sympathy_and_support': 5, 'injured_or_dead_people': 6, 
                            'caution_and_advice': 7, 'other_relevant_information': 8,
                            'not_humanitarian': 9}

    def valid_inputs(self, sentence1, label):
        return len(sentence1) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split('\t')
                try:
                    text_idx = header.index("tweet_text")
                    id_idx = header.index("tweet_id") if "tweet_id" in header else -1
                    label_idx = header.index("class_label") if "class_label" in header else -1
                except ValueError:
                    # If headers are missing standard names, fall back to pandas or just fail gracefully
                    print(f"Warning: Could not find expected columns in {path}. Header: {header}")
                    return []

                num_cols = len(header)
                
                for line_num, line in enumerate(f, start=2):
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split('\t')
                    
                    # Handle malformed lines by merging excess into tweet_text
                    if len(parts) > num_cols:
                        excess = len(parts) - num_cols
                        # The text content is spread from text_idx to text_idx + 1 + excess
                        # We join them with a space (assuming the tab was a separator in the text)
                        text_content = " ".join(parts[text_idx : text_idx + 1 + excess])
                        
                        # Reconstruct the parts list
                        new_parts = parts[:text_idx] + [text_content] + parts[text_idx + 1 + excess:]
                        parts = new_parts
                    
                    # Just in case it's still weird, check if we have enough parts for the required indices
                    # We only strictly require text and label (if label is needed)
                    required_indices = [i for i in [text_idx, label_idx, id_idx] if i != -1]
                    if not required_indices:
                        continue
                        
                    max_req_idx = max(required_indices)
                    if len(parts) <= max_req_idx:
                        # Not enough columns to reach the data we need
                        continue
                        
                    # (Removed strict len(parts) < num_cols check to allow lines with missing trailing/optional columns)

                    try:
                        guid = parts[id_idx] if id_idx != -1 else ""
                        sentence1 = parts[text_idx]
                        label_str = parts[label_idx] if label_idx != -1 else None

                        if self.valid_inputs(sentence1, label_str):
                            label = int(self.label_map[label_str])
                            samples.append((sentence1, label, guid))
                    except Exception:
                        continue

        except Exception as e:
            print(f"Error loading {path}: {e}")

        return samples

class CrisisMMDINFProcessor:
    """
    Loads CrisisMMD informative classification from a Hugging Face
    saved-to-disk directory (e.g., .../crisismmd2inf_dataset/train).

    It emits 4-tuples that your TextDataset expects for *pair* classification:
    (sentence1, sentence2, label_int, guid)
    where:
    - sentence1 = tweet_text
    - sentence2 = event_name  <-- added per request
    - label_int ∈ {0,1} (not_informative=0, informative=1)
    - guid = tweet_id (fallback to image_id if missing)

    NOTE: The pair gets encoded as:
    [CLS] sentence1 [SEP] sentence2 [SEP]
    which your encode_pair_inputs() already implements. No other code changes needed.
    """

    def __init__(self, label_field="label_text"):
        # Prefer text-only supervision for a text model; change to "label_text_image"
        # if you want the joint (text+image) supervision target instead.
        self.label_field = label_field
        # Map string labels -> ints. If your stored labels are already ints, we’ll cast below.
        self.label_map = {"not_informative": 0, "informative": 1}

    def valid_inputs(self, s1, label):
        # s1 must exist and label must be 0/1; s2 (event) can be empty string if missing.
        return (s1 is not None) and (len(s1) > 0) and (label in (0, 1))

    def _to_int_label(self, raw):
        # Accept ints (0/1) or strings ("informative"/"not_informative")
        if isinstance(raw, (int, float)) and int(raw) in (0, 1):
            return int(raw)
        return self.label_map.get(str(raw))

    def load_samples(self, path):
        # `path` points to a split folder produced by Dataset.save_to_disk()
        ds = load_from_disk(path)

        samples = []
        for ex in ds:
            # sentence1: tweet text
            s1 = ex.get("tweet_text")

            # # sentence2: event context (can be "")
            # # s2 = ex.get("event_name") or ""
            # s2 = ""

            # choose label source (prefer text-only; fallback to overall label)
            raw_label = ex.get(self.label_field)
            if raw_label is None:
                raw_label = ex.get("label")

            label = self._to_int_label(raw_label)

            # stable-ish ID for debugging/traceability
            guid = ex.get("tweet_id", [])

            if self.valid_inputs(s1, label):
                samples.append((s1, label, guid))

        return samples

class FEVERProcessor:
    """Data loader for QQP."""

    def __init__(self):
        self.label_map = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        # guid_check = []
        # label_check = []
        #new_file = open("./data/FEVER/sym_test_v1.txt","w")
        #new_file = open("./data/FEVER/train.jsonl","w")
        #new_data = []
        df = pd.read_json(path, lines=True)
        for i, (_, line) in enumerate(df.iterrows()):
            #line['id'] = line['id']#int(i)
            #new_data.append(dict(line))
            if "unique_id" in line:
                guid = line["unique_id"]
            else:
                guid = line["id"]

            sentence1 = line["claim"]
            try:
                sentence2 = line["evidence"]
                label = line["gold_label"]
            except:
                sentence2 = line["evidence_sentence"]
                label = line["label"]
            label = self.label_map[label]
            label = int(label)
            samples.append((sentence1, sentence2, label, guid))
        return samples

class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[7]
                    sentence2 = row[8]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples

class SNLITESTProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    # guid = row[0]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

##HANS
class HANSProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                sentence1 = row[5]
                sentence2 = row[6]
                label = row[0]
                if label == 'non-entailment':
                    label = 'contradiction'
                label = self.label_map[label]
                samples.append((sentence1, sentence2, label, []))
        return samples



class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

class MNLITESTProcessor:
    """Data loader for MNLI."""
    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    # ## Stress Test Eval
                    # sentence1 = row[1]
                    # sentence2 = row[2]
                    # label = row[0]

                    # # RTE
                    # if label == 'contradiction':
                    #     label = 0
                    # elif label == 'neutral':
                    #     label = 0 
                    # else:
                    #     label = 1
                    # samples.append((sentence1, sentence2, label, []))
                    sentence1 = row[5]
                    sentence2 = row[6]
                    label = row[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples


class SICKProcessor():
    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 2, 'neutral': 1}
        self.label_list = [0,1,2]

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_list


    def load_samples(self, path):
        samples = []
        e,c,n = 0,0,0
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #try:
                guid = row[0]
                sentence1 = row[1]
                sentence2 = row[2]
                try:
                    label = int(row[3])
                except:
                    label = self.label_map[row[3]]
                if self.valid_inputs(sentence1, sentence2, label):
                    samples.append((sentence1, sentence2, label, guid))
                # except:
                #     pass
        return samples



class RTEProcessor():
    def __init__(self):
        self.label_map = {'entailment': 1, 'not_entailment':0}
        
    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map


    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #try:
                guid = row[0]
                sentence1 = row[1]
                sentence2 = row[2]
                label = row[3]
                if self.valid_inputs(sentence1, sentence2, label):
                    label = int(self.label_map[label])
                    samples.append((sentence1, sentence2, label, guid))
                # except:
                #     pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        for item in data:
            row =  pp.commaSeparatedList.parseString(item).asList()
            try:
                guid = row[0]
                context = row[4]
                start_ending = row[5]
                endings = row[7:11]
                label = int(row[-1])
                samples.append((context, start_ending, endings, label, guid))
            except:
                pass
        return samples



class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label, []))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor'](event=args.event) if args.task == 'HumAID' else globals()[f'{args.task}Processor']()



def smoothing_label(target, smoothing):
    """Label smoothing"""
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob


class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor, num_instances=None, augment=False):
        # print(path)
        # if path is not None:
        self.samples = processor.load_samples(path)
        self.unlabeled = False
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, i):
    #     res = self.cache.get(i, None)
    #     if res is None:
    #         sample = self.samples[i]
    #         if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB','SICK','RTE','FEVER','HANS','CrisisMMDINF', 'HumAID'): # and not self.unlabeled:
    #             sentence1, sentence2, label, guid = sample
    #             input_ids, segment_ids, attention_mask = encode_pair_inputs(
    #                 sentence1, sentence2
    #             )
    #             label_id = encode_label(label)
    #             res = ((input_ids, segment_ids, attention_mask, guid, [sentence1+' [SEP] ' +sentence2]), label_id)
    #         elif args.task in ('SWAG', 'HellaSWAG'):
    #             if self.unlabeled:
    #                 context, ending_start, endings = sample
    #                 guid = -1
    #             else:
    #                 context, ending_start, endings, label, guid = sample
    #             input_ids, segment_ids, attention_mask = encode_mc_inputs(
    #                 context, ending_start, endings
    #             )
    #             label_id = encode_label(label)
    #             res = ((input_ids, segment_ids, attention_mask, guid), label_id)
    #         self.cache[i] = res
    #     return res

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]

            if args.task in ('SNLI','MNLI','QQP','MRPC','TwitterPPDB','SICK','RTE','FEVER','HANS'):
                sentence1, sentence2, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_pair_inputs(sentence1, sentence2)
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask, guid, [sentence1 + ' [SEP] ' + sentence2]), label_id)

            elif args.task in ('CrisisMMDINF', 'HumAID'):
                # single-sentence case
                # expected sample format: (text, label, guid)
                # if your sample layout differs, adjust the unpacking accordingly
                text, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_single_inputs(text)
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask, guid, [text]), label_id)

            elif args.task in ('SWAG', 'HellaSWAG'):
                if self.unlabeled:
                    context, ending_start, endings = sample
                    guid = -1
                    label = None
                else:
                    context, ending_start, endings, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_mc_inputs(context, ending_start, endings)
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask, guid), label_id)

            self.cache[i] = res
        return res



class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)

        hidden = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden, n_classes)

        if args.task in ('SWAG', 'HellaSWAG'):
            self.n_choices = -1

    def forward(self, input_ids, segment_ids, attention_mask, unlabeled=False):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = input_ids.size(1)
            self.n_choices = n_choices
            input_ids = input_ids.view(-1, input_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        transformer_params = {
            'input_ids': input_ids,
            # BERT uses token_type_ids; RoBERTa/BERTweet do not.
            'token_type_ids': (segment_ids if args.model in ('bert-base-uncased','bert-large-uncased') else None),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        #if args.consistency_learning or args.noisy_label:
        if args.ssl:
            return transformer_outputs
        else:
            # Always use CLS from last_hidden_state for logits
            last_hidden = transformer_outputs[0]        # [B, L, H]
            cls_output  = last_hidden[:, 0]             # [B, H]
            logits = self.classifier(cls_output)
            if args.task in ('SWAG', 'HellaSWAG'):
                logits = logits.view(-1, self.n_choices)
            return logits

prev_mean1 = cuda(torch.tensor(1.,dtype=torch.float))#, cuda(torch.tensor(1.,dtype=torch.float))
prev_std1 = cuda(torch.tensor(1.,dtype=torch.float))#, cuda(torch.tensor(1.,dtype=torch.float))

def take_cls(feat):
    """Return CLS/<s> features as [B, H] for any of:
    - [B, L, H]  last_hidden_state
    - [B, H]     already CLS
    - [H]        single example vector
    - HF BaseModelOutput (tuple-like)
    """
    if isinstance(feat, (tuple, list)):  # HF outputs
        feat = feat[0]  # last_hidden_state

    if feat.dim() == 3:          # [B, L, H]
        feat = feat[:, 0]        # -> [B, H]
    elif feat.dim() == 1:        # [H]
        feat = feat.unsqueeze(0) # -> [1, H]
    elif feat.dim() != 2:        # anything else is unexpected
        raise ValueError(f"Unexpected CLS tensor shape: {tuple(feat.shape)}")

    return feat  # guaranteed [B, H]


def train(d1,d2=None,aug=None,epoch=0):
    """Fine-tunes pre-trained model on training set."""
    global prev_mean1, prev_std1
    model.train()
    train_loss = 0.
    if args.ssl:
        d1_loader = load(d1,args.batch_size,True)
        d2_loader = tqdm(load(d2,args.unlabeled_batch_size,False))
        optimizer = AdamW(adamw_params(model),lr=args.learning_rate,eps=1e-8)
        alpha = 0.4
        lam = np.random.beta(alpha,alpha)
        high_count, low_count = 0,0
        high_per_iter, low_per_iter = [], []
        discard_per_iter = []
        unlabeled_not_used_count = 0
        total_num = []
        average = []
        discard_ratio = args.batch_size*0.1
        pbar = tqdm(total=len(d2_loader))
        pbar2 = tqdm(total=len(d2_loader))
        pbar3 = tqdm(total=len(d2_loader))
        pbar4 = tqdm(total=len(d2_loader))


        # Removed manual artifact files — wandb handles artifacting now.
        for i, (dataset1, dataset2) in enumerate(zip(cycle(d1_loader),d2_loader)):
            optimizer.zero_grad()
            inputs1, labels1 = dataset1
            inputs2, true_labels2 = dataset2
            guid = list(inputs1[3])
            original_unlabeled = inputs2[4][0]
            discard_count = 0 
            smoothing_val = 0.3

            if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','CrisisMMDINF', 'HumAID'):
                output1 = model(inputs1[0],inputs1[1],inputs1[2], unlabeled = False)[0]
                output2 = model(inputs2[0],inputs2[1],inputs2[2], unlabeled = True)[0]
                #logits1 = model.classifier(output1[:,0])
            elif args.task in ('SWAG'):
                output1 = model(inputs1[0],inputs1[1],inputs1[2], unlabeled = False)[1]
                output2 = model(inputs2[0],inputs2[1],inputs2[2], unlabeled = True)[1]
                #logits1 = model.classifier(output1)
            #if args.task in ('SWAG'): 
            #    logits1 = logits1.view(-1, model.n_choices)

            if args.multigpus:
                logits1 = model.module.classifier(output1[:,0])
            else:
                logits1 = model.classifier(output1[:,0])
            loss1 = criterion(logits1,labels1)

            if output1.shape[0] != output2.shape[0]:
                min_idx = min(output1.shape[0],output2.shape[0])
                output1 = output1[:min_idx,:]
                output2 = output2[:min_idx,:]
                true_labels2 = true_labels2[:min_idx]

            if args.pseudo_label_by_normalized:
                ## Moment injection to unlabeled features
                unlabeled_mean = torch.mean(output2,dim=1) 
                labeled_mean = torch.mean(output1,dim=1)
                unlabeled_std = torch.std(output2,dim=1)
                labeled_std = torch.std(output1,dim=1)
                output1 = output1[:,0]
                output2 = output2[:,0]
                
                output2_perturb = (output2 - unlabeled_mean)/unlabeled_std *labeled_std + labeled_mean
                logits2 = model.classifier(output2_perturb)
                if args.task in ('SWAG'): logits2 = logits2.view(-1,model.n_choices)
            else:
                output2_perturb = output2[:,0]
                if args.multigpus:
                    logits2 = model.module.classifier(output2[:,0])
                else:
                    logits2 = model.classifier(output2[:,0])

            ## Pseudo Label Generation
            # ALWAYS use CLS for both labeled and unlabeled
            cls1 = take_cls(output1)      # [B1, H]
            cls2 = take_cls(output2)      # [B2, H]
            H    = cls1.shape[-1]

            # logits for unlabeled CLS
            logits2 = (model.module.classifier(cls2) if args.multigpus else model.classifier(cls2))
            if args.task in ('SWAG'):
                logits2 = logits2.view(-1, model.n_choices)

            # soft labels with sharpening
            tmp_labels2 = F.softmax(logits2, dim=-1)
            if args.sharpening:
                tmp_labels2 = tmp_labels2 ** (1 / args.T)
                tmp_labels2 = tmp_labels2 / tmp_labels2.sum(dim=1, keepdim=True)

            verifier_prob, verifier_label = torch.max(tmp_labels2, dim=-1)
            original_prob, original_idx, original_true_labels2 = verifier_prob, verifier_label, true_labels2

            # allocate buffers in [*, H] (NO seq_len dim anywhere)
            mismatch_outputs = cuda(torch.zeros([int(cls2.shape[0]), H]))
            mismatch_labels  = cuda(torch.zeros([int(cls2.shape[0]), n_classes]))
            filtered_outputs = cuda(torch.zeros([int(cls2.shape[0]), H]))
            filtered_labels  = cuda(torch.zeros([int(cls2.shape[0])], dtype=torch.long))
            filtered_prob    = cuda(torch.zeros_like(verifier_prob))
            usage_check      = cuda(torch.full((int(cls2.shape[0]),), -1, dtype=torch.long))

            mismatch_idx, filtered_idx = 0, 0
            discard_count = 0

            # split matched vs mismatched (store CLS features only)
            for u_idx in range(verifier_prob.shape[0]):
                if verifier_label[u_idx] == true_labels2[u_idx]:
                    filtered_outputs[filtered_idx] = cls2[u_idx]                      # [H]
                    filtered_labels[filtered_idx]  = true_labels2[u_idx]
                    filtered_prob[filtered_idx]    = verifier_prob[u_idx]
                    usage_check[u_idx]             = 1
                    filtered_idx += 1
                else:
                    discard_count += 1
                    mismatch_outputs[mismatch_idx] = cls2[u_idx]                      # [H]
                    # one-hot only the true class prob from tmp_labels2
                    mismatch_labels[mismatch_idx, true_labels2[u_idx]] = tmp_labels2[u_idx][true_labels2[u_idx]]
                    mismatch_idx += 1

            # finalize trimmed tensors
            mismatch_outputs = mismatch_outputs[:mismatch_idx]  # [M, H]
            mismatch_labels  = mismatch_labels[:mismatch_idx]   # [M, C]
            filtered_outputs = filtered_outputs[:filtered_idx]  # [U, H]
            filtered_labels  = filtered_labels[:filtered_idx]   # [U]
            filtered_prob    = filtered_prob[:filtered_idx]

            # manual artifacting removed (wandb handles artifacts)

            # set avg threshold (global mean or your per-class variant)
            avg_prob = torch.mean(filtered_prob) if filtered_prob.numel() > 0 else torch.tensor(0., device=filtered_prob.device)

            # split matched into high/low by confidence (still CLS only)
            low_output  = cuda(torch.zeros([int(filtered_outputs.shape[0]), H]))
            high_output = cuda(torch.zeros([int(filtered_outputs.shape[0]), H]))
            high_true_labels = cuda(torch.zeros([int(filtered_outputs.shape[0])], dtype=torch.long))
            low_true_labels  = cuda(torch.zeros([int(filtered_outputs.shape[0])], dtype=torch.long))

            low_idx = high_idx = 0
            for k in range(filtered_outputs.shape[0]):
                if filtered_prob[k] >= avg_prob:
                    high_output[high_idx]     = filtered_outputs[k]
                    high_true_labels[high_idx]= filtered_labels[k]
                    high_idx += 1
                else:
                    low_output[low_idx]       = filtered_outputs[k]
                    low_true_labels[low_idx]  = filtered_labels[k]
                    low_idx += 1

            high_output     = high_output[:high_idx]           # [Hh, H]
            high_true_labels= high_true_labels[:high_idx]      # [Hh]
            low_output      = low_output[:low_idx]             # [Hl, H]
            low_true_labels = low_true_labels[:low_idx]        # [Hl]

            # ======== MIXUP pieces (sizes must match) ========

            # MixUp against low-confidence matched set (paper-style) by default
            # choose the smaller K across the two sources to ensure shapes match
            smoothing_val = 0.3
            if args.mixup:
                # choose which unlabeled pool to mix with
                if args.high_mixup:
                    pool_feat  = high_output
                    pool_y_int = high_true_labels
                else:
                    pool_feat  = low_output
                    pool_y_int = low_true_labels

                # label smoothing for the chosen pool (one-hot)
                pool_y = smoothing_label(pool_y_int, smoothing_val) if pool_feat.shape[0] > 0 else cuda(torch.zeros([0, n_classes]))

                # determine K that both sides can supply
                M_pool = pool_feat.shape[0]
                B_lab  = cls1.shape[0]
                K      = min(M_pool, B_lab)

                if K > 0:
                    # random K from each
                    idx_pool = torch.randperm(M_pool, device=cls1.device)[:K]
                    idx_lab  = torch.randperm(B_lab,  device=cls1.device)[:K]

                    labeled_part     = cls1[idx_lab]                         # [K, H]
                    labeled_onehot   = smoothing_label(labels1[idx_lab], smoothing_val)  # [K, C]
                    pool_part        = pool_feat[idx_pool]                   # [K, H]
                    pool_onehot      = pool_y[idx_pool]                      # [K, C]

                    # do the actual mix
                    mixup_output = labeled_part * lam + pool_part * (1 - lam)            # [K, H]
                    mixup_label  = labeled_onehot * lam + pool_onehot * (1 - lam)        # [K, C]

                    # classify the mixed features
                    mixup_output = (model.module.classifier(mixup_output)
                                    if args.multigpus else model.classifier(mixup_output))
                    if args.task in ('SWAG'):
                        mixup_output = mixup_output.view(-1, model.n_choices)
                    mixup_loss = torch.mean(torch.sum(-mixup_label * torch.log_softmax(mixup_output, dim=-1), dim=-1))
                else:
                    mixup_loss = cuda(torch.tensor(0.))
            else:
                mixup_loss = cuda(torch.tensor(0.))

            # ======== Discard-MixUp (mismatched vs labeled), sizes must match ========

            M = mismatch_outputs.shape[0]
            B = cls1.shape[0]
            K2 = min(M, B)

            if K2 > 0:
                idx_m = torch.randperm(M, device=cls1.device)[:K2]
                idx_l = torch.randperm(B, device=cls1.device)[:K2]

                labeled_sel     = cls1[idx_l]                                   # [K2, H]
                labeled_onehot2 = smoothing_label(labels1[idx_l], smoothing_val)# [K2, C]
                mis_sel         = mismatch_outputs[idx_m]                        # [K2, H]
                mis_onehot      = mismatch_labels[idx_m]                         # [K2, C]

                discardMixUp      = labeled_sel * lam + mis_sel * (1 - lam)      # [K2, H]
                discardMixUpLabels= labeled_onehot2 * lam + mis_onehot * (1 - lam)  # [K2, C]

                discardMixUp = (model.module.classifier(discardMixUp)
                                if args.multigpus else model.classifier(discardMixUp))
                discardMixUpLoss = torch.mean(torch.sum(-discardMixUpLabels * torch.log_softmax(discardMixUp, dim=-1), dim=-1))
            

            
            output2 = filtered_outputs[:filtered_idx]
            true_labels2 = filtered_labels[:filtered_idx]
            verifier_prob = filtered_prob[:filtered_idx]
            verifier_label = filtered_labels[:filtered_idx]

            # manual artifacting removed (wandb handles artifacts)
            discard_per_iter.append(discard_count)
            avg_prob = torch.mean(verifier_prob)
            # manual artifacting removed (wandb handles artifacts)
            ## When using median model confidence
            #avg_prob = torch.median(prob)
            ## When using fixed high threshold value
            # avg_prob = 0.9
            low_output = cuda(torch.zeros([int(output2.shape[0]), H])) 
            high_output = cuda(torch.zeros([int(output2.shape[0]), H]))
            high_true_labels, low_true_labels = cuda(torch.tensor([0 for _ in range(int(verifier_label.shape[0]))])),cuda(torch.tensor([0 for _ in range(int(verifier_label.shape[0]))]))#cuda(torch.zeros([int(idx.shape[0]/2)])),cuda(torch.zeros([int(idx.shape[0]/2)]))
            low_idx, high_idx, c_idx = 0,0,0
            high_inputs, low_inputs = [],[]
            for k in range(0,output2.shape[0]):
                if verifier_prob[k] >= avg_prob:
                    high_output[high_idx] = output2[k]
                    high_true_labels[high_idx] = verifier_label[k]#cuda(torch.tensor(idx[k].data.tolist(),dtype=torch.int64))#true_labels2[k]#idx[k]#true_labels2[k]
                    # manual artifacting removed (wandb handles artifacts)
                    high_count += 1
                    high_idx += 1
                else:
                    low_output[low_idx] = output2[k]
                    low_true_labels[low_idx] = verifier_label[k]#true_labels2[k]
                    # manual artifacting removed (wandb handles artifacts)
                    low_count += 1
                    low_idx += 1
            high_true_labels = high_true_labels[:high_idx]
            high_output = high_output[:high_idx]
            low_true_labels = low_true_labels[:low_idx]
            low_output = low_output[:low_idx]
            
            if args.mixup:
                if args.rand_mixup:
                    select_idx = torch.randperm(int(output2.shape[0]/2))
                    to_be_mixed_output = output2[select_idx]
                    to_be_mixed_label = labels2[select_idx]
                    rand_high_idx = 0
                    for i in range(output2.shape[0]):
                        if i not in select_idx:
                            high_output[rand_high_idx] = output2[i]
                            high_labels[rand_high_idx] = labels2[i]
                            rand_high_idx += 1
                else:
                    if args.high_mixup:
                        select_idx = torch.randperm(high_output.shape[0])
                        to_be_mixed_output = high_output
                        to_be_mixed_label = high_labels
                    else:
                        low_labels = smoothing_label(low_true_labels,smoothing_val)
                        select_idx = torch.randperm(low_output.shape[0])
                        to_be_mixed_output = low_output
                        to_be_mixed_label = low_labels
                    output1 = output1[select_idx]
                    labels1 = labels1[select_idx]
                    labels1_onehot = smoothing_label(labels1,smoothing_val)

                    if not args.pseudo_label_by_normalized:
                        # PSEUDO_LABEL PATCH
                        cls1 = take_cls(output1)
                        K = min(cls1.size(0), to_be_mixed_output.size(0))
                        idx_lab = torch.randperm(cls1.size(0))[:K]
                        idx_pool = torch.randperm(to_be_mixed_output.size(0))[:K]
                        mixup_output = cls1[idx_lab] * lam + to_be_mixed_output[idx_pool] * (1 - lam)
                    else:
                        mixup_output = output1 * lam + to_be_mixed_output * (1-lam)

                    mixup_label = labels1_onehot * lam + to_be_mixed_label * (1-lam)
                if args.multigpus:
                    high_logits = model.module.classifier(high_output)
                else:
                    high_logits = model.classifier(high_output)
                high_labels = smoothing_label(high_true_labels,smoothing_val)
                loss2 = torch.mean(torch.sum(-high_labels*torch.log_softmax(high_logits,dim=-1),dim=-1))# *checkterm)
                if args.multigpus:
                    mixup_output = model.module.classifier(mixup_output)
                else:
                    mixup_output = model.classifier(mixup_output)
                if args.task in ('SWAG'): mixup_output = mixup_output.view(-1, model.n_choices)
                mixup_loss = torch.mean(torch.sum(-mixup_label * torch.log_softmax(mixup_output, dim=-1), dim=-1))
                if low_labels.shape[0] == 0:
                    mixup_loss = cuda(torch.tensor(0.))
                if mismatch_labels.shape[0] == 0:
                    discardMixUpLoss = cuda(torch.tensor(0.))
                if high_labels.shape[0] == 0:
                    loss2 = cuda(torch.tensor(0.))
                loss = loss1 + loss2 +  args.mixup_loss_weight*mixup_loss + discardMixUpLoss
            else:
                if args.multigpus:
                    high_logits = model.module.classifier(high_output)
                else:
                    high_logits = model.classifier(high_output)
                high_labels = smoothing_label(high_true_labels,0.3)
                loss2 = torch.mean(torch.sum(-high_labels*torch.log_softmax(high_logits,dim=-1),dim=-1))# *checkterm)
                
                loss = loss1 + loss2 + discardMixUpLoss
                mixup_loss = torch.tensor(0.)
            
            train_loss += loss.item()
            pbar.set_description(f"supervised loss = {(loss1.item())/(i+1):.6f}")
            pbar.update(1)
            # d1_loader.set_description(f"mixup train loss = {(mixup_loss.item() / (i+1)):.6f}")
            pbar2.set_description(f"mixup train loss = {(mixup_loss.item() / (i+1)):.6f}")
            pbar2.update(1)
            d2_loader.set_description(f'unsupervised loss = {loss2.item()/(i+1):.10f}')

            pbar3.set_description(f"total train loss = {(train_loss / (i+1)):.6f}")
            pbar3.update(1)
            pbar4.set_description(f"Discard Mixup Loss = {discardMixUpLoss.item()/(i+1):.6f}")
            pbar4.update(1)
            loss.backward()
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        
        # manual artifacting removed (wandb handles artifacts)
        return train_loss / (len(d1_loader)+len(d2_loader))
    else:
        train_loader = tqdm(load(d1, args.batch_size, True))
        optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
        for i, dataset in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = dataset
            guid = inputs[3]
            logit = model(inputs[0],inputs[1],inputs[2])
            prob = F.softmax(logit,dim=-1)
            max_prob,pred_label = torch.max(prob,dim=-1)
            loss = criterion(logit,labels)
            train_loss += loss.item()
            train_loader.set_description(f'train loss = {(train_loss / (i+1)):.6f}')
            loss.backward()
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates model; returns loss, accuracy(%), macro-F1."""
    model.eval()
    eval_loss = 0.0
    y_true, y_pred = [], []
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, batch in enumerate(eval_loader):
        with torch.no_grad():
            inputs, labels = batch
            output = model(inputs[0], inputs[1], inputs[2])
            if args.ssl:
                if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','HANS','CrisisMMDINF','HumAID'):
                    output = model.module.classifier(output[0][:,0]) if args.multigpus else model.classifier(output[0][:,0])
                elif args.task in ('SWAG'):
                    output = model.classifier(output[1]).view(-1, model.n_choices)

            y_pred.extend(output.argmax(dim=-1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            loss = criterion(output, labels)

        eval_loss += loss.item()
        eval_loader.set_description(f"eval loss = {(eval_loss/(i+1)):.6f}")

    eval_acc = accuracy_score(y_true, y_pred) * 100.0
    eval_f1  = f1_score(y_true, y_pred, average='macro')
    return (eval_loss / len(eval_loader)), eval_acc, eval_f1

def predict(dataset, return_probs=True):
    """
    Run model inference over a dataset. Returns:
      y_true: list[int]
      y_pred: list[int]
      y_conf: list[float] (if return_probs)
      guids:  list[Any]   (best-effort)
      texts:  list[str]   (best-effort)
    """
    model.eval()
    y_true, y_pred, y_conf = [], [], []
    guids, texts = [], []

    loader = tqdm(load(dataset, args.batch_size, False))
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            out = model(inputs[0], inputs[1], inputs[2])

            # -- same logits path as evaluate() --
            if args.ssl:
                if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','HANS','CrisisMMDINF','HumAID'):
                    logits = model.module.classifier(out[0][:,0]) if args.multigpus else model.classifier(out[0][:,0])
                elif args.task in ('SWAG'):
                    logits = model.classifier(out[1]).view(-1, model.n_choices)
            else:
                logits = out

            probs = F.softmax(logits, dim=-1)
            pred  = probs.argmax(dim=-1)

            # collect labels/preds
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            if return_probs:
                y_conf.extend(probs.max(dim=-1).values.cpu().tolist())

            # collect GUIDs (best-effort; may be [] in some loaders)
            batch_guids = inputs[3]
            if isinstance(batch_guids, (list, tuple)):
                # ensure simple scalars/strings or fallback to ""
                guids.extend([g if isinstance(g, (str, int)) else "" for g in batch_guids])
            else:
                try:
                    guids.extend(batch_guids.cpu().numpy().tolist())
                except Exception:
                    guids.extend([""] * logits.size(0))

            # collect text (you packaged each example text as a 1-element list)
            batch_texts = inputs[4] if len(inputs) > 4 else [""] * logits.size(0)
            
            # Fix for DataLoader collation of list-wrapped strings:
            # The loader returns [('t1', 't2', ...)] or [['t1', 't2', ...]] (len=1)
            # We must unpack this to iterate over the actual strings.
            if isinstance(batch_texts, list) and len(batch_texts) == 1 and isinstance(batch_texts[0], (list, tuple)):
                batch_texts = batch_texts[0]

            flat_texts = []
            for t in batch_texts:
                if isinstance(t, list) and len(t) > 0:
                    flat_texts.append(t[0])
                elif isinstance(t, str):
                    flat_texts.append(t)
                else:
                    flat_texts.append("")
            texts.extend(flat_texts)

    return y_true, y_pred, y_conf, guids, texts


model = cuda(Model())
if args.multigpus:
    model = nn.DataParallel(model)
processor = select_processor()

# [wandb-B] track model and define metrics
wandb.watch(model, log="all", log_freq=100)
wandb.define_metric("epoch")
# Epoch time-series
for key in ["train/loss", "val/loss", "val/acc", "val/f1", "epoch_time_min"]:
    wandb.define_metric(key, step_metric="epoch")

wandb.define_metric("train_loss", step_metric="epoch")
wandb.define_metric("eval_loss", step_metric="epoch")
wandb.define_metric("eval_acc", step_metric="epoch")
wandb.define_metric("eval_f1", step_metric="epoch")
wandb.define_metric("subrun")
wandb.define_metric("best_dev_f1", step_metric="subrun")
wandb.define_metric("best_epoch", step_metric="subrun")


if args.model == 'vinai/bertweet-base':
    # BERTweet: RoBERTa-like tokenizer; normalization helps on noisy Twitter text.
    # Some HF versions expose `normalization`; guard it to stay compatible.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, normalization=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

criterion = nn.CrossEntropyLoss()

if args.task == 'MNLI':
    test_processor = MNLITESTProcessor()
    # test_processor = globals()[f'MNLITESTProcessor']()
    m_path = "./data/MNLI/multinli_0.9_dev_matched.txt"
    match_dataset = TextDataset(m_path, test_processor)
    mm_path = "./data/MNLI/multinli_0.9_dev_mismatched.txt"
    mismatch_dataset = TextDataset(mm_path,test_processor)



# ===============================================================
# Averaged sweep-compatible training: one W&B run per config
# ===============================================================
set_nums = [args.set_num]
seeds = [67]

macro_f1_scores, acc_scores = [], []
dev_f1_scores = []
subrun_idx = 0

# Accumulate all predictions for a single artifact



for set_num in set_nums:
    for seed in seeds:
        sub_run_start = time.time()
        args.set_num = set_num
        args.seed = seed
        print(f"\n=== Running set_num={set_num}, seed={seed} ===\n")

        # Update dataset paths
        paths = get_paths(event, lbcl, args.set_num)
        args.dev_path = paths["dev_path"]
        args.test_path = paths["test_path"]
        args.labeled_train_path = paths["train_labeled_path"]
        args.unlabeled_train_path = paths["train_unlabeled_path"]
        args.output_path = fr"{paths['vmatch_out']}/preds_set{set_num}_seed{seed}.json"
        args.ckpt_path = fr"{paths['vmatch_out']}/model_set{set_num}_seed{seed}.pt"

        # Reinitialize model + tokenizer for each seed
        model = cuda(Model())
        if args.multigpus:
            model = nn.DataParallel(model)
        processor = select_processor()

        if args.model == 'vinai/bertweet-base':
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, normalization=True)
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)

        criterion = nn.CrossEntropyLoss()

        if args.ssl:
            d1 = TextDataset(args.labeled_train_path, processor)
            d2 = TextDataset(args.unlabeled_train_path, processor)
            print(f'labeled train samples = {len(d1)}')
            print(f'unlabeled train samples = {len(d2)}')
        else:
            train_dataset = TextDataset(args.train_path, processor)
            print(f'train samples = {len(train_dataset)}')

        if args.dev_path:
            dev_dataset = TextDataset(args.dev_path, processor)
            print(f'dev samples = {len(dev_dataset)}')
        if args.test_path:
            test_dataset = TextDataset(args.test_path, processor)
            print(f'test samples = {len(test_dataset)}')


        # === Training loop ===
        best_dev_f1 = -float('inf')
        best_epoch = -1
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            # --- training step ---
            if args.ssl:
                train_loss = train(d1=d1, d2=d2, epoch=epoch)
            else:
                train_loss = train(d1=train_dataset, epoch=epoch)

            # --- evaluation step ---
            eval_loss, eval_acc, eval_f1 = evaluate(dev_dataset)

            # --- keep best checkpoint (by dev macro-F1) ---
            if eval_f1 > best_dev_f1:
                best_dev_f1 = eval_f1
                best_epoch = epoch
                torch.save(model.state_dict(), args.ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Uncomment for full artifact control
            # if args.artifact_mode in ('periodic','all'):
            #     save_this_epoch = (args.artifact_mode == 'all') or (epoch % args.artifact_every == 0)
            #     if save_this_epoch:
            #         tmp_ckpt = fr"{paths['vmatch_out']}/epoch{epoch}_set{set_num}_seed{seed}.pt"
            #         torch.save(model.state_dict(), tmp_ckpt)
            #         art_name = f"{args.task}-{event}-lb{lbcl}-set{set_num}-seed{seed}-epoch{epoch}"
            #         art = wandb.Artifact(art_name, type="model-epoch",
            #                             metadata={"epoch": epoch, "set_num": set_num, "seed": seed})
            #         art.add_file(tmp_ckpt, name="model.pt")
            #         wandb.log_artifact(art, aliases=[f"epoch-{epoch}"])
            #         if not args.keep_local_ckpt:
            #             try: os.remove(tmp_ckpt)
            #             except: pass


            # --- log + print ---
            elapsed = time.time() - start_time
            print(f"[set {set_num} seed {seed}] epoch {epoch}/{args.epochs} "
                f"| train={train_loss:.4f} | val_loss={eval_loss:.4f} "
                f"| val_acc={eval_acc:.2f} | val_f1={eval_f1:.4f} "
                f"| time={elapsed/60:.2f} min")

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": eval_loss,
                "val/acc": eval_acc,
                "val/f1": eval_f1,
                "epoch_time_min": elapsed / 60,
                "set_num": set_num,
                "seed": seed
            })


        # After epochs for this sub-run:
        wandb.log({
            "subrun": subrun_idx,
            "set_num": set_num,
            "seed": seed,
            "best_dev_f1": best_dev_f1,
            "best_epoch": best_epoch,
        })
        dev_f1_scores.append(best_dev_f1)
        subrun_idx += 1

        # === Test evaluation ===
        model.load_state_dict(torch.load(args.ckpt_path))
        test_loss, test_acc_pct, test_f1 = evaluate(test_dataset)
        
        # Cleanup model checkpoint if requested
        if not args.keep_local_ckpt:
            try: os.remove(args.ckpt_path)
            except: pass

        # Collect raw predictions (and confidence) for artifacting
        y_true, y_pred, y_conf, guids, texts = predict(test_dataset, return_probs=True)

        run_predictions = []
        for yt, yp, cf, gid, txt in zip(y_true, y_pred, y_conf, guids, texts):
            run_predictions.append({
                "set_num": set_num,
                "seed": seed,
                "guid": gid,
                "text": txt,
                "label": int(yt),
                "pred": int(yp),
                "conf": float(cf)
            })

        # --- Save Run Artifact ---
        if len(run_predictions) > 0:
            # Save as JSONL
            preds_path_json = fr"{paths['vmatch_out']}/preds_set{set_num}_seed{seed}.jsonl"
            with open(preds_path_json, "w", encoding="utf-8") as f:
                for p in run_predictions:
                    f.write(json.dumps(p) + "\n")
            
            # Save as CSV
            preds_path_csv = fr"{paths['vmatch_out']}/preds_set{set_num}_seed{seed}.csv"
            try:
                pd.DataFrame(run_predictions).to_csv(preds_path_csv, index=False)
            except Exception as e:
                print(f"Warning: Could not save CSV predictions: {e}")

            print(f"Logging predictions artifact: {len(run_predictions)} rows.")
            pred_art = wandb.Artifact(f"preds-{args.task}-{event}-lb{lbcl}-set{set_num}-seed{seed}", type="predictions")
            pred_art.add_file(preds_path_json, name="preds.jsonl")
            if os.path.exists(preds_path_csv):
                pred_art.add_file(preds_path_csv, name="preds.csv")
            wandb.log_artifact(pred_art)
            
            # Cleanup
            if not args.keep_local_ckpt:
                try: 
                    os.remove(preds_path_json)
                    if os.path.exists(preds_path_csv):
                        os.remove(preds_path_csv)
                except: pass

        # Preds saved to cumulative list, artifacting happens after loop
        if not args.keep_local_ckpt:
            try: os.remove(args.ckpt_path)
            except: pass


        # keep your aggregates consistent with earlier code
        macro_f1_scores.append(test_f1)
        acc_scores.append(test_acc_pct / 100.0)  # your dev acc is %, your old test acc was 0..1

        print(f"✅ Completed set={set_num}, seed={seed} | F1={test_f1:.4f} | "
            f"Acc={test_acc_pct/100.0:.4f}")

        wandb.log({
            "subrun": subrun_idx - 1,  # same index as this sub-run
            "set_num": set_num,
            "seed": seed,
            "sub_macro-F1": test_f1,
            "sub_acc": test_acc_pct / 100.0,
        })


# === After all runs, average results ===
mean_f1 = np.mean(macro_f1_scores)
std_f1  = np.std(macro_f1_scores)
mean_acc = np.mean(acc_scores)
mean_dev_f1 = np.mean(dev_f1_scores)
std_dev_f1  = np.std(dev_f1_scores)



wandb.log({
    "test_macro-F1": mean_f1,
    "test_macro-F1_std": std_f1,
    "test_eval_acc": mean_acc,
    "dev_macro-F1": mean_dev_f1,
    "dev_macro-F1_std": std_dev_f1
})

print(f"\n=== Averaged Results ===")
print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Mean Acc: {mean_acc:.4f}")



wandb.finish()