# from comet_ml import Experiment
import os
import sys
import gc
import random
import time
import torch
import argparse
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    RobertaTokenizer,
    BertTokenizer,
    AutoModelForMaskedLM,
    AutoProcessor,
    CLIPTokenizer,
    get_scheduler,
    logging as transformers_logging
)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import logging as lg
 
import csv


from dotenv import load_dotenv
load_dotenv()

import wandb
# Set WandB local directory to /tmp to prevent workspace clutter
os.environ['WANDB_DIR'] = '/tmp'

# Local imports
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import delete_saved_models, log_message, str2bool
from data_processor import TextDataset
from models import RoBERTa, BERT, DeBERTa, RoBERTaLarge, TransformerModel, BERTweet, ClassifyCLIP
from loss import SmoothCrossEntropyLoss
# from gen_init_weights import WeightGenerator
# from co_training_parallel import CoTrainer
# from fine_tune_models import DualModelTrainer
from trainer_classes import WeightGenerator, CoTrainer, DualModelTrainer
import data_utils

import os
import numpy as np
import pandas as pd

# Constants
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LEN = 300 
CLIP_MAX_LEN = 77
EPOCH_PATIENCE = 5

# Dataset configurations

LABELED_SAMPLES = {
    'informative': [2, 10, 20, 40, 100, 200, 300, 400, 500, 1000],
    'humanitarian': [2, 10, 20, 40, 100, 200, 300, 400, 500],
    'humaid': [2, 10, 20, 40, 100, 200, 300, 400, 500]
}

NUM_CLASSES = {
    'informative': 2,
    'humanitarian': 5,
    'humaid': 10
}

# Model mapping for easier reference
HF_MODEL_MAPPING = {
    "phi-3": "Phi-3-medium-4k",
    "phi-3-128k": "Phi-3-medium-128k",
    "mistral-7b": "Mistral-7B-Instruct",
    "llama-3-8b": "Llama-3.1-8B",
    "llama-3-70b": "Llama-3.3-70B",
    "roberta": "roberta-base",
    "N/A": "N/A"
}

PLM_ID_MAPPING = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta-base": "microsoft/deberta-base",
    "deberta-large": "microsoft/deberta-large",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "bert-tweet": "vinai/bertweet-base",
    "clip": "openai/clip-vit-base-patch32"
}

few_shot_samples_per_class = {
    'informative': 2,
    'humanitarian': 5,
    'humaid': 10,
}

plm_ids = list(PLM_ID_MAPPING.keys())
llm_ids = list(HF_MODEL_MAPPING.keys())
datasets = list(LABELED_SAMPLES.keys())

# functions moved to data_utils.py


def create_dataloader(dataframe, tokenizer, dataset_name, batch_size, max_len):
    """Create a DataLoader for a given dataset."""
    dataset_obj = TextDataset(dataframe, tokenizer, max_len, dataset=dataset_name)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)


def initialize_models(num_classes, args):
    """Initialize models based on PLM type."""
    if "clip" == args.plm_id:
        print(f"Using CLIP model: {args.plm_id}")
        model_1 = ClassifyCLIP(num_classes=num_classes, single_modality=True, text_embed=True, image_embed=False)
        model_2 = ClassifyCLIP(num_classes=num_classes, single_modality=True, text_embed=True, image_embed=False)
    elif "bert-tweet" == args.plm_id:
        print(f"Using Bert Tweet model: {args.plm_id}")
        model_1 = BERTweet(num_classes=num_classes, args=args)
        model_2 = BERTweet(num_classes=num_classes, args=args)
    elif "roberta-base" == args.plm_id:
        print(f"Using RoBERTa model: {args.plm_id}")
        model_1 = RoBERTa(num_classes=num_classes, args=args)
        model_2 = RoBERTa(num_classes=num_classes, args=args)
    elif "bert-base" == args.plm_id:
        print(f"Using BERT model: {args.plm_id}")
        model_1 = BERT(num_classes=num_classes, args=args)
        model_2 = BERT(num_classes=num_classes, args=args)
    elif "deberta-base" == args.plm_id:
        print(f"Using DeBERTa model: {args.plm_id}")
        model_1 = DeBERTa(num_classes=num_classes, args=args)
        model_2 = DeBERTa(num_classes=num_classes, args=args)
    elif "roberta-large" == args.plm_id:
        print(f"Using RoBERTa model: {args.plm_id}")
        model_1 = RoBERTaLarge(num_classes=num_classes, args=args)
        model_2 = RoBERTaLarge(num_classes=num_classes, args=args)
    else:
        print(f"Model type {args.plm_id} not recognized. Defaulting to RoBERTa base.")
        model_1 = RoBERTa(num_classes=num_classes, args=args)
        model_2 = RoBERTa(num_classes=num_classes, args=args)
    return model_1, model_2


def setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=nn.CrossEntropyLoss):
    """Set up optimizers, schedulers and criterion for training."""
    criterion = criterion_class(reduction='none')
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']
    train_dataloader_1 = dataloaders['train_dataloader_1']
    train_dataloader_2 = dataloaders['train_dataloader_2']
    
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=learning_rate, weight_decay=training_params.get('weight_decay', 0.01))
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=learning_rate, weight_decay=training_params.get('weight_decay', 0.01))
    
    num_training_steps_1 = num_epochs * len(train_dataloader_1)
    num_training_steps_2 = num_epochs * len(train_dataloader_2)
    
    lr_scheduler_1 = get_scheduler(
        name="linear", 
        optimizer=optimizer_1, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_1
    )
    
    lr_scheduler_2 = get_scheduler(
        name="linear", 
        optimizer=optimizer_2, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_2
    )
    
    optimizer_params = {
        'criterion': criterion,
        'optimizer_1': optimizer_1,
        'optimizer_2': optimizer_2,
        'num_training_steps_1': num_training_steps_1,
        'num_training_steps_2': num_training_steps_2,
        'lr_scheduler_1': lr_scheduler_1,
        'lr_scheduler_2': lr_scheduler_2
    }
    
    return optimizer_params

def get_batch_size(dataset, plm_id):
    if plm_id == 'bert-base':
        return 24 if dataset not in ['swag', 'hellaswag'] else 8
    elif plm_id == 'roberta-base':
        return 24 if dataset not in ['swag', 'hellaswag'] else 8
    elif plm_id == 'deberta-base':
        return 16 if dataset not in ['swag', 'hellaswag'] else 4
    else:
        # default fallback
        return 8

def calculate_ece(y_true, y_pred, confidences, n_bins=10):
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(np.array(y_pred)[bin_mask] == np.array(y_true)[bin_mask])
            bin_conf = np.mean(np.array(confidences)[bin_mask])
            ece += (np.sum(bin_mask) / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

def evaluate_models(model_1, model_2, eval_dataloader, device_1, device_2):
    """Evaluate ensembled models on provided dataloader."""
    model_1.eval()
    model_2.eval()
    y_true = []
    y_pred = []
    confidences = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Process on first device
            batch_1 = {k: v.to(device_1) for k, v in batch.items()}
            outputs_1 = model_1(input_ids=batch_1['input_ids'], attention_mask=batch_1['attention_mask'])
            outputs_1 = outputs_1.logits if hasattr(outputs_1, 'logits') else outputs_1
            val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1)
            
            # Process on second device
            batch_2 = {k: v.to(device_2) for k, v in batch.items()}
            outputs_2 = model_2(input_ids=batch_2['input_ids'], attention_mask=batch_2['attention_mask'])
            outputs_2 = outputs_2.logits if hasattr(outputs_2, 'logits') else outputs_2
            val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1)
            
            # Ensemble predictions
            val_probs = val_probs_1.cpu() + val_probs_2.cpu()
            out_ensembled = torch.argmax(val_probs, dim=1)
            confidence = torch.max(val_probs, dim=1)[0]
            out_ensembled = out_ensembled.cpu().detach().numpy()
            confidence = confidence.cpu().detach().numpy()
            
            # Collect predictions and ground truth
            y_pred_batch = out_ensembled.tolist()
            y_true_batch = batch_1['labels'].cpu().numpy().tolist()
            confidences_batch = confidence.tolist()
            
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
            confidences.extend(confidences_batch)
    
    cur_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate ECE
    ece = calculate_ece(y_true, y_pred, confidences)
    
    return cur_f1, acc, ece

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Co-Training Script")
    parser.add_argument("--dataset", type=str,  choices=datasets, help="Dataset name")
    parser.add_argument("--labeled_sample_idx", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Index for labeled samples")
    parser.add_argument("--hf_model_id_short", type=str, choices=llm_ids, help="Short ID for the Hugging Face model")
    parser.add_argument("--seed", type=int, default=1234, choices=[1234, 4567, 8998], help="Random seed for reproducibility")
    parser.add_argument("--plm_id", type=str, default="roberta-base", choices=plm_ids, help="PLM (bert-base, roberta-base, deberta-base, etc.)")
    parser.add_argument("--pseudo_label_shot", type=int, default=0, help="Number of pseudo labeled samples")
    parser.add_argument("--few_shot", action="store_true", default=False, help="Use few-shot prompted pseudolabels.")
    parser.add_argument("--single_set", action="store_true", default=False, help="Use single training set for both models")
    parser.add_argument("--no_co_training", action="store_true", default=False, help="Disable co-training")
    parser.add_argument("--metric_combination", type=str, default='cv', choices=["cv", "cc"], help="Metric combination method")
    parser.add_argument("--exp_name", type=str, default="lg-cotr", help="Experiment name")
    parser.add_argument("--setup_local_logging", action="store_true", default=False, help="Setup local logging")
    parser.add_argument("--comet_ml", action="store_true", default=False, help="Use comet_ml for experiment tracking")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for experiment tracking")
    parser.add_argument("--use_correct_labels_only", type=str2bool, default=False, help="Use correct labels only")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3", help="Comma-separated list of CUDA device IDs to use (e.g., 0,1)")
    parser.add_argument("--imb_training", action="store_true", default=False, help="Use imbalanced training")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--pseudo_label_dir", type=str, default="./data/pseudo_labels", help="Path to pseudo-label directory")
    parser.add_argument("--event", type=str, default=None, help="Event name for humaid dataset")
    parser.add_argument("--lbcl", type=str, default=None, help="Labeled count per class string/int for humaid dataset")
    parser.add_argument("--set_num", type=str, default=None, help="Set number string for humaid dataset")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--epoch_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides default)")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Accumulation steps (overrides default)")
    args = parser.parse_args()
    
    #args.pseudo_label_shot = few_shot_samples_per_class[args.dataset] if args.few_shot else 0
    
    return args

def set_environment(args):
    """Set environment variables and random seeds."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    transformers_logging.set_verbosity_error()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device configuration
    if torch.cuda.device_count() >= 2:
        device_1 = torch.device("cuda:0")
        device_2 = torch.device("cuda:1")
    else:
        device_1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_2 = device_1
        
    return device_1, device_2

def setup_local_logging(args):
    """Set up logging to file and console."""
    if not args.setup_local_logging:
        return None
    
    log_dir = f"{ROOT}/output/{args.dataset}/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    output_log_path = os.path.join(log_dir, f"log_{args.saved_model_name_suffix}.txt")
    
    lg.basicConfig(
        filename=output_log_path,
        filemode='w',
        level=lg.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = lg.getLogger()
    return logger

def setup_comet_experiment(args):
    """Set up Comet ML experiment."""
    if not args.comet_ml:
        return None
    
    
    from comet_ml import Experiment
    experiment = Experiment(
        api_key="<Comet_api_key>",
        project_name="llmcot",
        workspace="YOUR_WANDB_WORKSPACE"
    )
    experiment.set_name(f"{args.dataset}_{args.saved_model_name_suffix}")
    return experiment

def setup_wandb_experiment(args):
    """Set up Wandb experiment."""
    if not hasattr(args, 'use_wandb') or not args.use_wandb:
        return None
    
    wandb.init(
        project="cotrain-hyperparameter-tuning",
        name=f"{args.dataset}_{args.saved_model_name_suffix}",
        config={
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "epoch_patience": args.epoch_patience,
            "dataset": args.dataset,
            "plm_id": args.plm_id,
            "seed": args.seed,
            "event": args.event,
            "lbcl": args.lbcl,
            "lbcl": args.lbcl,
            "set_num": args.set_num,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps
        }
    )
    return wandb

# load_dataset_helper moved to data_utils.py

def main():
    st = time.time()
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment and devices
    device_1, device_2 = set_environment(args)
    unique_devices = len(set([device_1, device_2]))
    
    # Determine model and dataset configurations
    # dataset = args.dataset
    N = LABELED_SAMPLES[args.dataset][args.labeled_sample_idx] // 2
    hf_model_name = HF_MODEL_MAPPING[args.hf_model_id_short]
    
    # Set pseudo_label_shot based on model
    if args.hf_model_id_short == "roberta":
        args.pseudo_label_shot = N * 2
    
    # Set up experiment name
    # args.exp_name = "lg-cotr"
    
    # Set up paths
    saved_model_name_suffix = f"_{args.exp_name}_{args.hf_model_id_short}_{args.pseudo_label_shot}_shot_{args.plm_id}_{N}_seed_{args.seed}".replace('/', '-')
    if args.dataset == "humaid":
        saved_model_name_suffix = f"_{args.exp_name}_{args.hf_model_id_short}_{args.pseudo_label_shot}_shot_{args.plm_id}_{N}_seed_{args.seed}_{args.event}_{args.lbcl}".replace('/', '-')
            
    args.saved_model_name_suffix = saved_model_name_suffix
    
    # Set up directories
    data_dir = args.data_dir
    saved_model_dir = f"{ROOT}/saved_models/{args.dataset}/{args.exp_name}"
    processed_dir = f"{ROOT}/processed/{args.dataset}/{args.hf_model_id_short}"
    # save_dir = os.path.join(processed_dir, f'N_{N}')
    
    args.saved_model_dir = saved_model_dir
    
    
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)
    
    # Set batch size based on dataset and args.plm_id
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    else:
        BATCH_SIZE = get_batch_size(args.dataset, args.plm_id)
    
    
    # Set up hyperparameters
    # Set MAX_LEN based on model
    max_len = CLIP_MAX_LEN if args.plm_id == "clip" else (130 if args.plm_id == "bert-tweet" else DEFAULT_MAX_LEN)

    hyper_params = {
        'BATCH_SIZE': BATCH_SIZE,
        'MAX_LEN': max_len,
    }
    
    # Create dataloaders
    # Initialize tokenizer
    if args.plm_id == "clip":
        tokenizer = CLIPTokenizer.from_pretrained(PLM_ID_MAPPING[args.plm_id])
    elif args.plm_id == "bert-tweet":
        # BERTweet requires use_fast=False to avoid token ID mismatches with the model vocabulary
        tokenizer = AutoTokenizer.from_pretrained(PLM_ID_MAPPING[args.plm_id], use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PLM_ID_MAPPING[args.plm_id])

    # Load datasets
    trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data = data_utils.load_dataset_helper(
        use_correct_labels_only=args.use_correct_labels_only,
        shots=args.pseudo_label_shot,
        task_name=args.dataset,
        data_dir=data_dir,
        pseudo_label_dir=args.pseudo_label_dir,
        event=args.event,
        lbcl=args.lbcl,
        set_num=args.set_num
    )
    
    dataloaders = {
        'train_dataloader_1': create_dataloader(trainingSet_1, tokenizer, args.dataset, BATCH_SIZE, max_len),
        'train_dataloader_2': create_dataloader(trainingSet_2, tokenizer, args.dataset, BATCH_SIZE, max_len),
        'val_dataloader': create_dataloader(validationSet, tokenizer, args.dataset, BATCH_SIZE, max_len),
        'test_dataloader': create_dataloader(testingSet, tokenizer, args.dataset, BATCH_SIZE, max_len),
        'auto_label_dataloader': create_dataloader(auto_labeled_data, tokenizer, args.dataset, BATCH_SIZE, max_len)
    }
    
    # Training parameters
    training_params = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'accumulation_steps': 1,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm
    }
    
    # Initialize models and Set up optimizers and criterion for initial weight generation
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)

    # Resize embeddings to match tokenizer length
    # This prevents CUDA device-side assertions for models with unmapped tokens (e.g. BERTweet)
    def resize_embeddings(model, tokenizer_len):
        resized = False
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(tokenizer_len)
            resized = True
            print(f"Resized model.bert embeddings to {tokenizer_len}")
        elif hasattr(model, 'transformer'):
            model.transformer.resize_token_embeddings(tokenizer_len)
            resized = True
            print(f"Resized model.transformer embeddings to {tokenizer_len}")
        elif hasattr(model, 'clip_model'):
            # CLIPTextModel inherits from PreTrainedModel, so it supports resize_token_embeddings
            model.clip_model.resize_token_embeddings(tokenizer_len)
            resized = True
            print(f"Resized model.clip_model embeddings to {tokenizer_len}")
        elif hasattr(model, 'align_model'):
             # AlignModel also inherits from PreTrainedModel
             model.align_model.resize_token_embeddings(tokenizer_len)
             resized = True
             print(f"Resized model.align_model embeddings to {tokenizer_len}")
        
        if not resized:
            print(f"WARNING: Could not find known attribute to resize embeddings for model type {type(model)}")
        
    resize_embeddings(model_1, len(tokenizer))
    resize_embeddings(model_2, len(tokenizer))

    optimizer_params = setup_optimization(model_1, model_2, dataloaders,training_params, criterion_class=nn.CrossEntropyLoss)
    
    
    # Generate initial weights
    log_message(message='Generating initial weights', args=args)
    generator = WeightGenerator(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params={'BATCH_SIZE': BATCH_SIZE, 'MAX_LEN': max_len, 'EPOCH_PATIENCE': args.epoch_patience},
        devices=(device_1, device_2),
        models=(model_1, model_2),
        auto_labeled_data=auto_labeled_data,
        # metric_combination='cv'
    )
    init_df = generator.generate_weights()
    
    # Add init_df to dataloaders
    dataloaders['init_df_dataloader'] = create_dataloader(init_df, tokenizer, args.dataset, BATCH_SIZE, max_len)

    # Cleanup WeightGenerator and initial models to free memory
    del generator
    del model_1
    del model_2
    del optimizer_params
    gc.collect()
    torch.cuda.empty_cache()
    
    # Re-initialize models for co-training and Set up optimizers with SmoothCrossEntropyLoss for co-training
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    resize_embeddings(model_1, len(tokenizer))
    resize_embeddings(model_2, len(tokenizer))
    optimizer_params = setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=SmoothCrossEntropyLoss)
    
    # Co-training
    log_message(message='Starting co-training', args=args)
    trainer = CoTrainer(
        args=args,
        models={'model_1': model_1, 'model_2': model_2},
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params={'BATCH_SIZE': BATCH_SIZE, 'MAX_LEN': max_len, 'EPOCH_PATIENCE': args.epoch_patience},
        devices=[device_1, device_2],
        init_df=init_df,
        # metric_combination='cv'
    )
    co_training_df = trainer.train()
    #save the co_training_df to a file
    co_training_df.to_csv(os.path.join(saved_model_dir, f'co_training_df{saved_model_name_suffix}.csv'), index=False)
    
    # print(co_training_df.columns)
    # print(co_training_df[['id', 'ori_label', 'gen_label','train_weights_1', 'train_weights_2', 'all_epoch_probabilities_1', 'all_epoch_probabilities_2']].head(10))
    # print(co_training_df.head(10))
    # time.sleep(5000)
    
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load co-trained models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    resize_embeddings(model_1, len(tokenizer))
    resize_embeddings(model_2, len(tokenizer))
    model_1_path = f'{saved_model_dir}/co_trained_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/co_trained_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    # Set up fine-tuning parameters
    training_params['num_epochs'] = args.num_epochs
    hyper_params['EPOCH_PATIENCE'] = args.epoch_patience
    
    # Set up optimizers for fine-tuning
    optimizer_params = setup_optimization(
        model_1, model_2, 
        dataloaders,
        training_params
    )
    
    # Fine-tune models
    log_message(message='Fine-tuning models', args=args)
    dual_trainer = DualModelTrainer(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=(device_1, device_2),
        models=(model_1, model_2)
    )
    dual_trainer.train()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load fine-tuned models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    resize_embeddings(model_1, len(tokenizer))
    resize_embeddings(model_2, len(tokenizer))
    model_1_path = f'{saved_model_dir}/final_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/final_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    model_1.to(device_1)
    model_2.to(device_2)
    
    # Evaluate models on Validation set
    val_dataloader = dataloaders['val_dataloader']
    val_f1, val_acc, val_ece = evaluate_models(model_1, model_2, val_dataloader, device_1, device_2)

    val_result_msg = (f"\n\nHf Model: {hf_model_name} PLM: {args.plm_id} Dataset: {args.dataset}, NumShots: {args.pseudo_label_shot}, "
                 f"N: {N} Validation SEED: {args.seed} F1: {val_f1:.4f}, "
                 f"Validation Accuracy: {val_acc:.4f}, ECE: {val_ece:.4f}")
    
    log_message(message=val_result_msg, args=args)

    if hasattr(args, 'wandb_exp') and args.wandb_exp:
        args.wandb_exp.log({"val_f1": val_f1, "val_accuracy": val_acc, "val_ece": val_ece})

    # Evaluate models on Test set
    test_dataloader = dataloaders['test_dataloader']
    test_f1, test_acc, test_ece = evaluate_models(model_1, model_2, test_dataloader, device_1, device_2)
    
    # Log and print final results
    result_msg = (f"\n\nHf Model: {hf_model_name} PLM: {args.plm_id} Dataset: {args.dataset}, NumShots: {args.pseudo_label_shot}, "
                 f"N: {N} Test SEED: {args.seed} F1: {test_f1:.4f}, "
                 f"Test Accuracy: {test_acc:.4f}, ECE: {test_ece:.4f}")
    
    log_message(message=result_msg, args=args)
    
    # Log to wandb if available
    if hasattr(args, 'wandb_exp') and args.wandb_exp:
        args.wandb_exp.log({"test_f1": test_f1, "test_accuracy": test_acc, "test_ece": test_ece})
    
    msg = f"\nTotal time taken: {time.time() - st:.2f} seconds"
    log_message(message=msg, args=args)


if __name__ == "__main__":
    #torch.cuda.set_device(1)
    main()

# python3 main_bharani.py --dataset humanitarian --labeled_sample_idx 5 --seed 1234 --plm_id bert-tweet --setup_local_logging --comet_ml


# python3 main.py --dataset yelp_review --labeled_sample_idx 0 --hf_model_id_short llama-3-8b --seed 1234 --plm_id roberta-base --imb_training 
