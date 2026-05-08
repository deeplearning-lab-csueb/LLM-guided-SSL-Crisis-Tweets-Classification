import argparse
import logging
import os
import sys
import pandas as pd
import wandb

from transformers import AutoConfig, AutoTokenizer

# Add UST folder to sys.path so we can import its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
ust_dir = os.path.join(os.path.dirname(current_dir), "ust")
sys.path.append(ust_dir)

from aum_mixup_st import train_model_st_with_aummixup, CustomDataset

logger = logging.getLogger('AUM-ST-Mixup')
if sys.version_info >= (3, 8):
    logging.basicConfig(level=logging.INFO, force=True)
else:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel(logging.INFO)

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

label_to_id = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", help="path of the disaster directory containing train, test and unlabeled data files")
    parser.add_argument("--event", type=str, help="Event name (for HPO path resolution)")
    parser.add_argument("--lbcl", type=int, help="Labeled examples per class (for HPO path resolution)")
    parser.add_argument("--set_num", type=int, help="Set number (for HPO path resolution)")
    parser.add_argument("--train_file", nargs="?", type=str, default="S1T_5", help="train file" )
    parser.add_argument("--dev_file", nargs="?", type=str, default="S1V_5", help="train file")
    parser.add_argument("--unlabeled_file", nargs="?", type=str, default="S1U", help="train file")
    parser.add_argument("--aum_save_dir", nargs="?", type=str, default="data/AUM", help="Aum save directory")
    parser.add_argument("--seq_len", nargs="?", type=int, default=128, help="sequence length")
    parser.add_argument("--sup_batch_size", nargs="?", type=int, default=16, help="batch size for fine-tuning base model")
    parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=64, help="batch size for self-training on pseudo-labeled data")
    parser.add_argument("--sample_size", nargs="?", type=int, default=1800, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
    parser.add_argument("--unsup_size", nargs="?", type=int, default=1000, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
    parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
    parser.add_argument("--T", nargs="?", type=int, default=7, help="number of masked models for uncertainty estimation")
    parser.add_argument("--alpha", nargs="?", type=float, default=0.1, help="hyper-parameter for confident training loss")
    parser.add_argument("--sup_epochs", nargs="?", type=int, default=18, help="number of epochs for fine-tuning base model")
    parser.add_argument("--unsup_epochs", nargs="?", type=int, default=12, help="number of self-training iterations")
    parser.add_argument("--N_base", nargs="?", type=int, default=3, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
    parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="vinai/bertweet-base", help="teacher model checkpoint to load pre-trained weights")
    parser.add_argument("--results_file", nargs="?", default="result.txt", help="file name")
    parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
    parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for hidden layer of teacher model")
    parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.3, help="dropout probability for attention layer of teacher model")
    parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")
    parser.add_argument("--temp_scaling", nargs="?", type=lambda x: str(x).lower() == "true", default=False, help="temp scaling")
    parser.add_argument("--label_smoothing", nargs="?", type=float, default=0.0, help="label smoothing factor")
    
    args = vars(parser.parse_args())

    if os.environ.get("DEBUG"):
        logger.info("!!! DEBUG MODE ENABLED !!!")
        args["sup_epochs"] = 1
        args["unsup_epochs"] = 1
        args["N_base"] = 1
        args["sample_size"] = 100
        args["unsup_size"] = 50

    if args["event"] and args["lbcl"] and args["set_num"]:
        event = args["event"]
        lbcl = args["lbcl"]
        set_num = args["set_num"]
        disaster_name = event
        
        base_data_path = "data/humaid"
        if not os.path.exists(base_data_path) and os.path.exists("../data/humaid"):
            base_data_path = "../data/humaid"
            
        train_path = f"{base_data_path}/anh_4o/sep/{lbcl}lb/{set_num}/labeled.tsv"
        unlabeled_path = f"{base_data_path}/anh_4o/sep/{lbcl}lb/{set_num}/unlabeled.tsv"
        dev_path = f"{base_data_path}/joined/dev.tsv"
        test_path = f"{base_data_path}/joined/test.tsv"

        use_hpo_paths = True
        run_name = f"{event}_{lbcl}lb_set{set_num}_aum_mixup"
        
        wandb.init(
            project="humaid_aum_mixup_st_hpo",
            entity="YOUR_WANDB_ENTITY",
            name=run_name,
            config=args,
            reinit=True
        )
    else:
        if not args["disaster"]:
            parser.error("At least --disaster OR (--event, --lbcl, --set_num) must be provided.")
            
        disaster_name = args["disaster"]
        train_file = args["train_file"]
        train_path = "data/" + disaster_name + "/labeled_" + train_file + ".tsv"
        dev_path = "data/" + disaster_name + "/" + disaster_name + "_dev.tsv"
        test_path = "data/" + disaster_name + "/" + disaster_name + "_test.tsv"
        unlabeled_path = "data/" + disaster_name + "/unlabeled_" + train_file + ".tsv"
        
        use_hpo_paths = False
        run_name = f"{disaster_name}_legacy_aum_mixup"

    event_name = args.get("event")
    if event_name and event_name in EVENT_CLASS_MAPPING:
        labels = sorted(EVENT_CLASS_MAPPING[event_name])
        label_to_id.update({l: i for i, l in enumerate(labels)})
        logger.info(f"Using {len(labels)} classes for event {event_name}")
    else:
        default_labels = [
            "caution_and_advice", "displaced_people_and_evacuations", 
            "infrastructure_and_utility_damage", "injured_or_dead_people", 
            "missing_or_found_people", "not_humanitarian", 
            "other_relevant_information", "requests_or_urgent_needs", 
            "rescue_volunteering_or_donation_effort", "sympathy_and_support"
        ]
        label_to_id.update({l: i for i, l in enumerate(default_labels)})
        logger.info("Using default 10 classes.")

    model_dir = run_name if use_hpo_paths else disaster_name

    cfg = AutoConfig.from_pretrained(args["pt_teacher_checkpoint"], token=os.environ.get("HF_TOKEN"))
    cfg.hidden_dropout_prob = args["hidden_dropout_prob"]
    cfg.attention_probs_dropout_prob = args["attention_probs_dropout_prob"]

    tokenizer = AutoTokenizer.from_pretrained(args["pt_teacher_checkpoint"], token=os.environ.get("HF_TOKEN"))

    def get_dataset_smart(path, tokenizer, labeled=True, filter_event=None):
        if not os.path.exists(path):
            if path.startswith("../") and not os.path.exists(path):
                path = path[3:]
        
        print(f"Loading data from {path}...")
        df = pd.read_csv(path, sep='\t')
        
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
                labels_list.append(-1)
                
            if 'tweet_id' in row:
                ids_list.append(row['tweet_id'])
            else:
                ids_list.append(i)
            
        return CustomDataset(text_list, labels_list, ids_list, tokenizer, labeled=labeled)

    if use_hpo_paths:
        ds_train = get_dataset_smart(train_path, tokenizer, labeled=True, filter_event=disaster_name)
        ds_unlabeled = get_dataset_smart(unlabeled_path, tokenizer, labeled=False, filter_event=disaster_name)
        ds_dev = get_dataset_smart(dev_path, tokenizer, labeled=True, filter_event=disaster_name)
        ds_test = get_dataset_smart(test_path, tokenizer, labeled=True, filter_event=disaster_name)
    else:
        ds_train = get_dataset_smart(train_path, tokenizer, labeled=True)
        ds_unlabeled = get_dataset_smart(unlabeled_path, tokenizer, labeled=False)
        ds_dev = get_dataset_smart(dev_path, tokenizer, labeled=True)
        ds_test = get_dataset_smart(test_path, tokenizer, labeled=True)

    logger.info(f"Loaded labeled set size: {len(ds_train)}")
    logger.info(f"Loaded unlabeled set size: {len(ds_unlabeled)}")
    logger.info(f"Loaded dev set size: {len(ds_dev)}")
    logger.info(f"Loaded test set size: {len(ds_test)}")

    train_model_st_with_aummixup(
        ds_train, ds_dev, ds_test, ds_unlabeled,
        args["pt_teacher_checkpoint"], cfg, model_dir, args["aum_save_dir"], len(label_to_id.keys()),
        sup_batch_size=args["sup_batch_size"],
        unsup_batch_size=args["unsup_batch_size"],
        unsup_size=args["unsup_size"],
        sample_size=args["sample_size"],
        sample_scheme=args["sample_scheme"],
        T=args["T"],
        alpha=args["alpha"],
        sup_epochs=args["sup_epochs"],
        unsup_epochs=args["unsup_epochs"],
        N_base=args["N_base"],
        dense_dropout=args["dense_dropout"],
        attention_probs_dropout_prob=args["attention_probs_dropout_prob"],
        hidden_dropout_prob=args["hidden_dropout_prob"],
        results_file=args["results_file"],
        temp_scaling=args["temp_scaling"],
        ls=args["label_smoothing"],
        run_name=run_name,
        token=os.environ.get("HF_TOKEN")
    )
