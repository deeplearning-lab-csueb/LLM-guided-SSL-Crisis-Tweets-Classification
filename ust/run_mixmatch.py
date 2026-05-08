from distutils.command.config import config
from sklearn.utils import shuffle

import argparse
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import transformers

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from custom_dataset import CustomDataset_tracked, CustomDataset
from ust import train_mixmatch
import wandb


# logging
logger = logging.getLogger('UST')
if sys.version_info >= (3, 8):
    logging.basicConfig(level=logging.INFO, force=True)
else:
    # Fallback for older python if needed, though force=True is cleaner
    logging.basicConfig(level=logging.INFO)
    # Clear existing handlers if any
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)

logger.setLevel(logging.INFO)

GLOBAL_SEED = 67
logger.info ("Global seed {}".format(GLOBAL_SEED))

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

# label map will be set dynamically based on event
label_to_id = {}

def get_dataset(path, tokenizer, labeled=True, event=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Use pandas for efficient reading and filtering
    df = pd.read_csv(path, sep='\t')
    
    if event:
        if 'event' in df.columns:
            df = df[df['event'] == event]
        # If 'event' column is missing, we assume the file is already specific to the event (like the labeled training files)
        # or we just can't filter. 
        # But for joined dev/test files, this is critical.
    
    text_list = []
    labels_list = []
    ids_list = []
    
    # Iterate over filtered dataframe
    for i, row in df.iterrows():
        if pd.isna(row['tweet_text']):
            continue
        text_list.append(row['tweet_text'])
        # Use the global label_to_id which should be set by now
        labels_list.append(label_to_id[row['class_label']])
        ids_list.append(row['tweet_id'])
        
    dataset = CustomDataset_tracked(text_list, labels_list, ids_list, tokenizer, labeled=labeled)       
    return dataset



if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--disaster", help="path of the disaster directory containing train, test and unlabeled data files")
	parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
	parser.add_argument("--event", type=str, help="Event name (for HPO path resolution)")
	parser.add_argument("--lbcl", type=int, help="Labeled examples per class (for HPO path resolution)")
	parser.add_argument("--set_num", type=int, help="Set number (for HPO path resolution)")
	parser.add_argument("--train_file", nargs="?", type=str, default="S1T_5", help="train file" )
	parser.add_argument("--dev_file", nargs="?", type=str, default="S1V_5", help="train file")
	parser.add_argument("--unlabeled_file", nargs="?", type=str, default="S1U", help="train file")
	parser.add_argument("--aum_save_dir", nargs="?", type=str, default="AUM0", help="Aum save directory")
	parser.add_argument("--seq_len", nargs="?", type=int, default=128, help="sequence length")
	parser.add_argument("--sup_batch_size", nargs="?", type=int, default=16, help="batch size for fine-tuning base model")
	parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=64, help="batch size for self-training on pseudo-labeled data")
	parser.add_argument("--sample_size", nargs="?", type=int, default=1800, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
	parser.add_argument("--unsup_size", nargs="?", type=int, default=1000, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
	parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
	# parser.add_argument("--sup_labels", nargs="?", type=int, default=60, help="number of labeled samples per class for training and validation (total)")
	parser.add_argument("--T", nargs="?", type=int, default=7, help="number of masked models for uncertainty estimation")
	parser.add_argument("--alpha", nargs="?", type=float, default=0.75, help="hyper-parameter for confident training loss (Beta dist param in MixMatch)")
	parser.add_argument("--T_sharpen", nargs="?", type=float, default=0.5, help="Sharpening temperature for MixMatch")

	# parser.add_argument("--valid_split", nargs="?", type=float, default=0.5, help="percentage of sup_labels to use for validation for each class")
	parser.add_argument("--sup_epochs", nargs="?", type=int, default=18, help="number of epochs for fine-tuning base model")
	parser.add_argument("--unsup_epochs", nargs="?", type=int, default=12, help="number of self-training iterations")
	parser.add_argument("--N_base", nargs="?", type=int, default=3, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="vinai/bertweet-base", help="teacher model checkpoint to load pre-trained weights")
	parser.add_argument("--results_file", nargs="?", default="result.txt", help="file name")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
	parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for hidden layer of teacher model")
	parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for attention layer of teacher model")
	parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")
	parser.add_argument("--temp_scaling", nargs="?", type=bool, default=False, help="temp scaling" )
	parser.add_argument("--label_smoothing", nargs="?", type=float, default=0.0, help="label smoothing factor")


	args = vars(parser.parse_args())
	logger.info(args)

    # --- Debug Mode ---
	if os.environ.get("DEBUG"):
		logger.info("!!! DEBUG MODE ENABLED !!!")
		args["sup_epochs"] = 1
		args["unsup_epochs"] = 1
		args["N_base"] = 1
		args["sample_size"] = 100
		args["unsup_size"] = 50

	# --- HPO / Data Path Logic ---
	# If HPO args are provided, resolve paths to standardized location
	if args["event"] and args["lbcl"] and args["set_num"]:
		event = args["event"]
		lbcl = args["lbcl"]
		set_num = args["set_num"]
		
		# Set disaster_name for fallback/logging
		disaster_name = event
		
		base_data_path = "../data/humaid"
		if not os.path.exists(base_data_path):
			# Fallback if running from a different root
			base_data_path = "data/humaid"
			
		# Construct standardized paths
		# Train: ../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/labeled.tsv
		train_path = f"{base_data_path}/anh_4o/sep/{lbcl}lb/{set_num}/labeled.tsv"
		
		# Unlabeled: ../data/humaid/anh_4o/sep/{lbcl}lb/{set_num}/unlabeled.tsv
		# (Assuming this exists based on structure, if not we might need a fallback or verify)
		unlabeled_path = f"{base_data_path}/anh_4o/sep/{lbcl}lb/{set_num}/unlabeled.tsv"
		
		# Dev/Test: ../data/humaid/joined/dev.tsv (need filtering by event)
		# NOTE: get_dataset reads the file directly. We need to handle filtering inside get_dataset or 
		# pass the event name to filter it if the file is the "joined" one. 
		# However, existing get_dataset just reads the whole file. 
		# For simplicity in this script, we assume specific dev/test files MIGHT exist or we use the joined one 
		# and trust the function to handle it? 
		# Actually, standard UST logic expects separate files per disaster. 
		# The joined file contains ALL events. We must filter.
		# But get_dataset implementation in this file reads everything.
		# Modification: We will point to the JOINED file, but we need to ensure the downstream code or 
		# the loading function handles filtering. 
		# The current get_dataset does NOT filter.
		# Let's check if there are split files available. 
		# supervised/bert_ft.py uses "../data/humaid/joined" and filters by event column.
		
		dev_path = f"{base_data_path}/joined/dev.tsv"
		test_path = f"{base_data_path}/joined/test.tsv"
		
		# We need to signal that these need filtering by 'event'
		use_hpo_paths = True
		
		# WandB Init for HPO
		# Use run_name to distinguish
		run_name = f"{event}_{lbcl}lb_set{set_num}_mixmatch"
		
		wandb.init(
			project="humaid_mixmatch_hpo",
			entity="YOUR_WANDB_ENTITY",
			name=run_name,
			config=args,
			reinit=True
		)
		
	else:
		# LEGACY MODE
		if not args["disaster"]:
			parser.error("At least --disaster OR (--event, --lbcl, --set_num) must be provided.")
			
		disaster_name = args["disaster"]
		train_file = args["train_file"]
		# Construct legacy paths
		train_path = "data/" + disaster_name + "/labeled_" + train_file + ".tsv"
		dev_path = "data/" + disaster_name + "/" + disaster_name + "_dev.tsv"
		test_path = "data/" + disaster_name + "/" + disaster_name + "_test.tsv"
		unlabeled_path = "data/" + disaster_name + "/unlabeled_" + train_file + ".tsv"
		
		use_hpo_paths = False
		run_name = f"{disaster_name}_legacy_mixmatch" # Placeholder

    # --- Populate label_to_id ---
	event_name = args.get("event")
	if event_name and event_name in EVENT_CLASS_MAPPING:
		labels = sorted(EVENT_CLASS_MAPPING[event_name])
		label_to_id.update({l: i for i, l in enumerate(labels)})
		logger.info(f"Using {len(labels)} classes for event {event_name}")
	else:
		# Fallback or legacy behavior
		default_labels = [
			"caution_and_advice", "displaced_people_and_evacuations", 
			"infrastructure_and_utility_damage", "injured_or_dead_people", 
			"missing_or_found_people", "not_humanitarian", 
			"other_relevant_information", "requests_or_urgent_needs", 
			"rescue_volunteering_or_donation_effort", "sympathy_and_support"
		]
		label_to_id.update({l: i for i, l in enumerate(default_labels)})
		logger.info("Using default 10 classes.")

	max_seq_length = args["seq_len"]
	sup_batch_size = args["sup_batch_size"]
	unsup_batch_size = args["unsup_batch_size"]
	unsup_size = args["unsup_size"]
	sample_size = args["sample_size"]
	if use_hpo_paths:
		model_dir = run_name
	else:
		model_dir = disaster_name
	aum_save_dir = args["aum_save_dir"]
	sample_scheme = args["sample_scheme"]
	# sup_labels = args["sup_labels"]
	T = args["T"]
	T_sharpen = args["T_sharpen"]
	alpha = args["alpha"]
	# valid_split = args["valid_split"]
	sup_epochs = args["sup_epochs"]
	unsup_epochs = args["unsup_epochs"]
	N_base = args["N_base"]
	pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
	do_pairwise = args["do_pairwise"]
	dense_dropout = args["dense_dropout"]
	attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
	hidden_dropout_prob = args["hidden_dropout_prob"]
	results_file_name = args["results_file"]
	# num_aug = args["num_aug"]
	train_file = args["train_file"]
	dev_file = args["dev_file"]
	unlabeled_file = args["unlabeled_file"]
	temp_scaling = args["temp_scaling"]
	label_smoothing = args["label_smoothing"]
	# test_disaster = args["test_disaster"]
	learning_rate = args["learning_rate"]


	cfg = AutoConfig.from_pretrained(pt_teacher_checkpoint, token=os.environ.get("HF_TOKEN"))
	cfg.hidden_dropout_prob = hidden_dropout_prob
	cfg.attention_probs_dropout_prob = attention_probs_dropout_prob

	tokenizer = AutoTokenizer.from_pretrained(pt_teacher_checkpoint, token=os.environ.get("HF_TOKEN"))
    

	# Override get_dataset to handle filtering if using HPO paths
	def get_dataset_smart(path, tokenizer, labeled=True, filter_event=None):
		if not os.path.exists(path):
			# Try fixing partial paths if running from root vs folder
			if path.startswith("../") and not os.path.exists(path):
				path = path[3:] # Remove ../
		
		print(f"Loading data from {path}...")
		df = pd.read_csv(path, sep='\t')
		
		# If filtering needed
		if filter_event:
			if 'event' in df.columns:
				print(f"Filtering dataset for event: {filter_event}")
				df = df[df['event'] == filter_event]
			else:
				print(f"Warning: filter_event={filter_event} requested but 'event' column not found in {path}")
				
		text_list = []
		labels_list = []
		ids_list = []
		for i, row in df.iterrows():
			if pd.isna(row['tweet_text']):
				continue
			text_list.append(row['tweet_text'])
			if labeled:
				labels_list.append(label_to_id[row['class_label']])
			else:
				labels_list.append(-1) # Placeholder
				
			# Handle ID col variations
			if 'tweet_id' in row:
				ids_list.append(row['tweet_id'])
			else:
				ids_list.append(i)
			
		if labeled:
			return CustomDataset_tracked(text_list, labels_list, ids_list, tokenizer, labeled=True)
		else:
			# For unlabeled, ust.py expects CustomDataset_tracked (or compatible)
			# The original code used CustomDataset for unlabeled?
			# Original: ds_unlabeled = get_dataset(..., False) -> CustomDataset_tracked
			# Wait, original get_dataset calls CustomDataset_tracked regardless of labeled=True/False
			# The tracked version is needed for ID tracking during uncertainty estimation
			return CustomDataset_tracked(text_list, labels_list, ids_list, tokenizer, labeled=False)

	# Load Datasets
	if use_hpo_paths:
		ds_train = get_dataset_smart(train_path, tokenizer, labeled=True, filter_event=disaster_name) # Unlabeled is filtered? No, train is specific file
		ds_unlabeled = get_dataset_smart(unlabeled_path, tokenizer, labeled=False, filter_event=disaster_name)
		ds_dev = get_dataset_smart(dev_path, tokenizer, labeled=True, filter_event=disaster_name)
		ds_test = get_dataset_smart(test_path, tokenizer, labeled=True, filter_event=disaster_name)

		logger.info(f"Loaded labeled set size: {len(ds_train)}")
		logger.info(f"Loaded unlabeled set size: {len(ds_unlabeled)}")
		logger.info(f"Loaded dev set size: {len(ds_dev)}")
		logger.info(f"Loaded test set size: {len(ds_test)}")
	else:
		# Use original get_dataset for legacy
		ds_train = get_dataset(train_path, tokenizer)
		ds_dev = get_dataset(dev_path, tokenizer)
		ds_test = get_dataset(test_path, tokenizer)
		ds_unlabeled = get_dataset(unlabeled_path, tokenizer, False)
     

	train_mixmatch(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, T_sharpen, sup_batch_size=sup_batch_size, unsup_batch_size=unsup_batch_size, unsup_size=unsup_size, sample_size=sample_size,
	            sample_scheme=sample_scheme, T=T, alpha=alpha, sup_epochs=sup_epochs, unsup_epochs=unsup_epochs, N_base=N_base, dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob,
				results_file=results_file_name, temp_scaling = temp_scaling, ls=label_smoothing, n_classes=len(label_to_id.keys()), token=os.environ.get("HF_TOKEN"))
