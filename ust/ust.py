"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time 
from torchmetrics.classification import MulticlassCalibrationError
import torch.nn.functional as F
import torch.nn as nn
from custom_dataset import CustomDataset, CustomDataset_tracked

import logging
import math
import numpy as np
import pandas as pd 
import os
import sampler
import torch
import json 
import statistics 
import random
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method
import wandb

logger = logging.getLogger('UST')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("The device is : ", device)

def multigpu(model):
    model = nn.DataParallel(model).to(device)
    return model 

class BertModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels=10, token=None):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, token=token)
        #self.model.classifier.dropout.p = 0.5
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)


    def forward(self, input_ids, token_type_ids, attention_mask, temperature_scaling=False):
        if temperature_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits /= temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


def mc_dropout_evaluate(model_dir, n_classes, pt_teacher_checkpoint, X_new_unlabeled_dataset, cfg, linear_dropout=0.5, T=30, token=None):

    cfg.return_dict = True

    model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
    state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    #print(len(X_new_unlabeled_dataset.text_list))

    y_T = np.zeros((T, len(X_new_unlabeled_dataset), n_classes))
    acc = None
    data_loader = torch.utils.data.DataLoader(
        X_new_unlabeled_dataset, batch_size=64, shuffle=False)   

    logger.info ("Yielding predictions looping over ...")
    with torch.no_grad():
        for i in tqdm(range(T)):
            y_pred = []
            for elem in data_loader:
                x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
                pred = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                y_pred.extend(pred.logits.cpu().numpy())

            #converting logits to probabilities
            y_T[i] = softmax(np.array(y_pred), axis=-1)
    logger.info (y_T)

    #compute mean
    y_mean = np.mean(y_T, axis=0)
    assert y_mean.shape == (len(X_new_unlabeled_dataset), n_classes)

    #compute majority prediction
    y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])
    assert y_pred.shape == (len(X_new_unlabeled_dataset),)

    #compute variance
    y_var = np.var(y_T, axis=0)
    assert y_var.shape == (len(X_new_unlabeled_dataset), n_classes)

    return y_mean, y_var, y_pred, y_T



def evaluate(model, n_classes, test_dataloader, criterion, batch_size, temp_scaling=False):
    full_predictions = []
    true_labels = []
    probabilities = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            results = torch.argmax(logits.logits, dim=1)
            prob = F.softmax(logits.logits.to('cpu'), dim=1)
            probabilities += list(prob)

            crt_loss += criterion(logits.logits, x['lbl']
                                ).cpu().detach().numpy()
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + list(elem['lbl'].cpu().detach().numpy())


    model.train()

    metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=10, norm='l1')
    metric = metric.to(device)

    preds = torch.stack(probabilities)
    preds = preds.to(device)

    orig = torch.tensor(true_labels, dtype=torch.float, device=device)

    ece_metric = metric(preds, orig).to(device)

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader), ece_metric



def	train_model_ust(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, sup_batch_size=16, unsup_batch_size=64, unsup_size=4096, sample_size=16384,
	            sample_scheme='easy_bald_class_conf', T=30, alpha=0.1, sup_epochs=20, unsup_epochs=25, N_base=10, dense_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3,
                results_file="", temp_scaling=False, ls=0.0, n_classes=10, learning_rate=5e-5, run_name="ust_experiment", token=None):

    start_time = time.time()
    patience = 5
    load_best = False
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling
    logger_dict["Label Smoothing"]= ls

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=sup_batch_size, shuffle=True)   
    validation_dataloader = torch.utils.data.DataLoader(
        ds_dev, batch_size=sup_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=128, shuffle=False)
    
    cfg.num_labels = 10
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1
    #run the base model n times with different initialization to select best base model based on validation loss
    best_f1_overall = 0
    crt_patience = 0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True
            model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(train_dataloader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
                    #print(logits)
                    loss = loss_fn(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(
                    model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)
                
                # Log base model training stats
                if wandb.run:
                    wandb.log({
                        "base_train_loss": loss.item(),
                        "base_dev_f1": f1_macro_validation,
                         "base_dev_ece": ece.item() if isinstance(ece, torch.Tensor) else ece,
                         "epoch": epoch
                    })

                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        # Ensure directory exists
                        if not os.path.exists("data/" + model_dir):
                            os.makedirs("data/" + model_dir)
                        #model.save_pretrained(model_dir+"/ust")
                        torch.save(model.state_dict(), "data/" + model_dir + "/pytorch_model.bin")
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 3:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    cfg.return_dict = True

    best_model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
    best_model.to(device)
    state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
    best_model.load_state_dict(state_dict)

    for epoch in range(unsup_epochs):


        if sample_size < len(ds_unlabeled):
            logger.info ("Evaluating uncertainty on {} number of instances sampled from {} unlabeled instances".format(sample_size, len(ds_unlabeled)))
            indices = np.random.choice(len(ds_unlabeled), min(sample_size, len(ds_unlabeled)), replace=False)
            X_new_unlabeled_dataset = ds_unlabeled.get_subset_dataset(indices)
        else:
            logger.info ("Evaluating uncertainty on {} number of instances".format(len(ds_unlabeled)))
            X_new_unlabeled_dataset = ds_unlabeled

        if 'uni' in sample_scheme:
            y_mean, y_var, y_T = None, None, None
        elif 'bald' in sample_scheme:
            y_mean, y_var, y_pred, y_T = mc_dropout_evaluate(model_dir, n_classes, pt_teacher_checkpoint, X_new_unlabeled_dataset, cfg, dense_dropout, T=T, token=token)
        else:
            logger.info ("Error in specifying sample_scheme: One of the 'uni' or 'bald' schemes need to be specified")
            exit(0)

        if 'soft' not in sample_scheme:
            copy_cfg.return_dict = True
            # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
            model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
            state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            data_loader = torch.utils.data.DataLoader(
                X_new_unlabeled_dataset, batch_size=128, shuffle=False)   
            y_pred = []
            with torch.no_grad():
                for elem in data_loader:
                    x = {key: elem[key].to(device)
                    for key in elem if key not in ['idx', 'weights']}
                    pred = model(
                    input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                    y_pred.extend(pred.logits.cpu().numpy())
            del model

            y_pred = np.array(y_pred)
            y_pred = np.argmax(y_pred, axis=-1).flatten()

        # sample from unlabeled set
        if 'conf' in sample_scheme:
            conf = True
        else:
            conf = False

        if 'bald' in sample_scheme and 'eas' in sample_scheme:
            f_ = sampler.sample_by_bald_easiness

        if 'bald' in sample_scheme and 'eas' in sample_scheme and 'clas' in sample_scheme:
            f_ = sampler.sample_by_bald_class_easiness

        if 'bald' in sample_scheme and 'dif' in sample_scheme:
            f_ = sampler.sample_by_bald_difficulty

        if 'bald' in sample_scheme and 'dif' in sample_scheme and 'clas' in sample_scheme:
            f_ = sampler.sample_by_bald_class_difficulty

        if 'uni' in sample_scheme:
            if unsup_size < len(X_new_unlabeled_dataset):
                indices = np.random.choice(len(X_new_unlabeled_dataset), unsup_size, replace=False)
                text_data = [X_new_unlabeled_dataset.text_list[i] for i in indices]
                X_new_unlabeled_dataset = CustomDataset(text_data, y_pred[indices], X_new_unlabeled_dataset.tokenizer, labeled=True)
            else:
                X_new_unlabeled_dataset = CustomDataset(X_new_unlabeled_dataset.text_list, y_pred, X_new_unlabeled_dataset.tokenizer, labeled=True)
        else:
            X_new_unlabeled_dataset = f_(X_new_unlabeled_dataset, y_mean, y_var, y_pred, unsup_size, 10, y_T=y_T)

        if not conf:
            logger.info ("Not using confidence learning.")
            X_new_unlabeled_dataset.weights = np.ones(len(X_new_unlabeled_dataset))
        else:
            logger.info ("Using confidence learning ")
            X_new_unlabeled_dataset.weights = -np.log(np.array(X_new_unlabeled_dataset.weights)+1e-10)*alpha

        copy_cfg.return_dict = True
        # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
        # model.classifier.dropout.p = 0.1
        # model.to(device)

        model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
        model.to(device)
        state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
        model.load_state_dict(state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        unsup_dataloader = torch.utils.data.DataLoader(
        X_new_unlabeled_dataset, batch_size=unsup_batch_size, shuffle=True)   
        loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
        loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

        data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10**5, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
        crt_patience = 0
        for epoch in range(sup_epochs):
            for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, unsup_dataloader)):
                cuda_tensors_supervised = {key: data_supervised[key].to(
                    device) for key in data_supervised if key not in ['idx', 'weights']}

                cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                    device) for key in data_unsupervised if key not in ['idx']}
                    
                merged_tensors = {}
                for k in cuda_tensors_supervised:
                    merged_tensors[k] = torch.cat((cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

                num_lb = cuda_tensors_supervised['input_ids'].shape[0]
                num_ulb = cuda_tensors_unsupervised['input_ids'].shape[0]

                optimizer.zero_grad()
                logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors['token_type_ids'], attention_mask=merged_tensors['attention_mask'])

                logits_lbls = logits.logits[:num_lb]
                logits_ulbl = logits.logits[num_lb:]

                loss_sup = loss_fn_supervised(logits_lbls, cuda_tensors_supervised['lbl'])
                loss_unsup = cuda_tensors_unsupervised['weights'] * loss_fn_unsupervised(logits_ulbl, cuda_tensors_unsupervised['lbl'])
                loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
                loss.backward()
                optimizer.step()

            f1_macro_validation, loss_validation, ece = evaluate(
                model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)
            print('Confident learning metrics', f1_macro_validation)

            if wandb.run:
                 wandb.log({
                    "train_loss": loss.item(),
                    "dev_macro-F1": f1_macro_validation,
                    "dev_ece": ece.item() if isinstance(ece, torch.Tensor) else ece,
                    "unsup_epoch": epoch
                })

            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation
                if best_f1 > best_f1_overall:
                    #model.save_pretrained(model_dir+"/ust")
                    torch.save(model.state_dict(), "data/" + model_dir + "/pytorch_model.bin")
                    best_f1_overall = best_f1
                print('New best macro validation', best_f1, 'Epoch', epoch)
                continue
        
            if crt_patience == 3:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

            crt_patience += 1

    copy_cfg.return_dict = True
    # model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/ust", config=copy_cfg)
    model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
    model.to(device)
    state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
    model.load_state_dict(state_dict)

    rel_file = model_dir + "-" + results_file

    f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size)

    logger.info ("Test macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best ST model"] = {}
    logger_dict["Best ST model"]["F1 before temp scaling"] = str(f1_macro_test)
    logger_dict["Best ST model"]["ECE before temp scaling"] = str(ece_metric)


    logger_dict["Best ST model"]["T before temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(validation_dataloader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], True)

                loss = loss_fn(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()

        # f1_macro_test, _, ece_metric = evaluate(model, test_dataloader, loss_fn, unsup_batch_size, True)

        f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size, True)

        logger_dict["Best ST model"]["F1 after temp scaling"] = str(f1_macro_test)
        logger_dict["Best ST model"]["ECE after temp scaling"] = str(ece_metric)

        logger_dict["Best ST model"]["T  after temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    with open("data/" + model_dir +"/"+ results_file + '.txt','w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))

    # Log Final Test Metrics to WandB
    if wandb.run:
        wandb.log({
            "test_macro-F1": f1_macro_test,
             "test_ece": ece_metric.item() if isinstance(ece_metric, torch.Tensor) else ece_metric,
             "test_loss": loss_test
        })

    # ---------------------------
    # Generate and Save Predictions Artifact
    # ---------------------------
    # We need to run inference one last time to get predictions for saving
    # evaluate() function aggregates predictions but doesn't return them in the structure we need usually
    # But wait, evaluate() returns f1, loss, ece. It computes full_predictions inside.
    # We can create a small helper or just re-run the prediction logic here cleanly.
    
    logger.info("Generating predictions for artifact...")
    model.eval()
    all_preds = []
    all_golds = []
    all_ids = []
    
    # We need to iterate over ds_test again to get IDs correctly if they aren't guaranteed in dataloader order (though they usually are)
    # The dataloader is shuffle=False, so order is preserved.
    # ds_test is a CustomDataset or CustomDataset_tracked. It has .ids_list
    
    # Re-instantiate dataloader just to be safe
    final_test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for elem in tqdm(final_test_dataloader):
            x = {key: elem[key].to(device) for key in elem if key not in ['idx']}
            logits = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            preds = torch.argmax(logits.logits, dim=1).cpu().detach().numpy()
            
            # Gold labels
            golds = elem['lbl'].cpu().detach().numpy()
            
            all_preds.extend(preds)
            all_golds.extend(golds)
            
    # IDs
    # ds_test.ids_list should correspond to the order
    all_ids = ds_test.idxes
    
    # Save to CSV
    # Ensure directory exists
    # artifact_dir = f"data/{model_dir}/artifacts"
    # os.makedirs(artifact_dir, exist_ok=True)
    # pred_file = f"{artifact_dir}/{run_name}.csv"
    
    # df_preds = pd.DataFrame({
    #     "id": all_ids,
    #     "gold": all_golds,
    #     "pred": all_preds
    # })
    # df_preds.to_csv(pred_file, index=False)
    # logger.info(f"Saved predictions to {pred_file}")
    
    if wandb.run:
        # Create a temporary file for the artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
             df_preds = pd.DataFrame({
                "id": all_ids,
                "gold": all_golds,
                "pred": all_preds
            })
             df_preds.to_csv(tmp.name, index=False)
             pred_file = tmp.name

        artifact = wandb.Artifact(name=f"{run_name}-preds", type="predictions")
        artifact.add_file(pred_file, name=f"{run_name}.csv") # naming it explicitly in artifact
        wandb.log_artifact(artifact)
        
        # Clean up temp file
        os.remove(pred_file)

    # ---------------------------
    # cleanup: Remove local files
    # ---------------------------
    try:
        model_path = "data/" + model_dir + "/pytorch_model.bin"
        if os.path.exists(model_path):
            os.remove(model_path)
            
        results_path = "data/" + model_dir +"/"+ results_file + '.txt'
        if os.path.exists(results_path):
            os.remove(results_path)
            
        # If model_dir was created just for this run (HPO), we might want to remove it if empty
        # But checking if it is empty is tricky if logs are there. 
        # For HPO run_name based dirs, it should be safe if we are sure only we used it.
        # But let's just delete the files we know we created for now to be safe.
        
        logger.info("Cleaned up local model and results files.")
        
    except Exception as e:
        logger.warning(f"Failed to clean up local files: {e}")


def	train_mixmatch(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, T_sharpen, 
                sup_batch_size=16, unsup_batch_size=64, unsup_size=4096, sample_size=16384,
	            sample_scheme="uniform",T=30, alpha=0.1, sup_epochs=20, unsup_epochs=25, N_base=10, dense_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3,
                results_file="", temp_scaling=False, ls=0.0, n_classes=10, token=None):

    load_best = False
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling
    logger_dict["Label Smoothing"]= ls

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=sup_batch_size, shuffle=True)   
    validation_dataloader = torch.utils.data.DataLoader(
        ds_dev, batch_size=sup_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=128, shuffle=False)
    
    cfg.num_labels = n_classes
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1
    #run the base model n times with different initialization to select best base model based on validation loss
    best_f1_overall = 0
    crt_patience = 0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True
            model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
            model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(train_dataloader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors['token_type_ids'], attention_mask=cuda_tensors['attention_mask'])
                    #print(logits)
                    loss = loss_fn(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(
                    model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)

                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        if not os.path.exists("data/" + model_dir):
                            os.makedirs("data/" + model_dir)
                        torch.save(model.state_dict(),"data/" + model_dir + "/pytorch_model.bin")
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 3:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    for epoch in range(unsup_epochs):
      
        # This section is to generate pseudo-labels for the unlabeled data using the teacher model and 
        # sharpen them using T_sharpen from the mixmatch algorithm. These pseudo-labels are then used 
        # for the mixup training in the next section
        copy_cfg.return_dict = True
        model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
        state_dict = torch.load("data/" + model_dir + "/pytorch_model.bin")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        data_loader = torch.utils.data.DataLoader(
            ds_unlabeled, batch_size=128, shuffle=False)   
        y_pred = []
        tweet_ids = []
        with torch.no_grad():
            for elem in data_loader:
                x = {key: elem[key].to(device)
                for key in elem if key not in ['weights']}
                pred = model(
                input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
                y_pred.extend(pred.logits.cpu().numpy())
                tweet_ids.extend(x['idx'].cpu())
        del model
        y_pred = np.array(y_pred)
        tweet_ids = np.array(tweet_ids)

        prob_df = pd.DataFrame(y_pred, columns=['Prob_' + str(i) for i in range(n_classes)])
        prob_df["tweet_id"] = tweet_ids

        # Group by tweet_id and calculate the average of the probabilities
        avg_probs_df = prob_df.groupby('tweet_id').mean().reset_index()

        # Divide the probabilities by T
        avg_probs_df[['Prob_' + str(i) for i in range(n_classes)]] = avg_probs_df[['Prob_' + str(i) for i in range(n_classes)]] / T_sharpen

        # Find the column with the maximum value for each row and extract the integer part
        avg_probs_df['Max_Prob_Column'] = avg_probs_df[['Prob_' + str(i) for i in range(n_classes)]].idxmax(axis=1)
        avg_probs_df['Label'] = avg_probs_df['Max_Prob_Column'].str.extract('(\d+)').astype(int)
        
        # Drop the helper column if desired
        avg_probs_df = avg_probs_df.drop(columns=['Max_Prob_Column'])

        unlabeled_df = pd.DataFrame()
        unlabeled_df["text"] = ds_unlabeled.text_list
        unlabeled_df["tweet_id"] = ds_unlabeled.idxes

        # Merge the two DataFrames on tweet_id
        merged_df = pd.merge(unlabeled_df , avg_probs_df[['tweet_id', 'Label']], on='tweet_id', how='left')
        

        # Now we have the pseudo-labeled dataset in merged_df which contains the text, tweet_id and the pseudo-labels. 
        # We can now use this dataset for mixup training along with the original labeled dataset

        model = BertModel(pt_teacher_checkpoint, num_labels=n_classes, token=token)
        model.to(device)
        state_dict = torch.load("data/" +model_dir + "/pytorch_model.bin")
        model.load_state_dict(state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
        model.train()

        psuedolabeled_dataset = CustomDataset_tracked(merged_df["text"], merged_df["Label"], merged_df["tweet_id"], ds_unlabeled.tokenizer, labeled = True)

        unsup_dataloader = torch.utils.data.DataLoader(
        psuedolabeled_dataset, batch_size=unsup_batch_size, shuffle=True)   
        loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
        loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

        data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10**5, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
        crt_patience = 0

        for epoch in range(sup_epochs):
            for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, unsup_dataloader)):
                cuda_tensors_supervised = {key: data_supervised[key].to(
                    device) for key in data_supervised if key not in ['idx', 'weights']}

                cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                    device) for key in data_unsupervised if key not in ['idx']}
                    
                merged_tensors = {}
                for k in cuda_tensors_supervised:
                    merged_tensors[k] = torch.cat((cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

                num_lb = cuda_tensors_supervised['input_ids'].shape[0]
                num_ulb = cuda_tensors_unsupervised['input_ids'].shape[0]

                optimizer.zero_grad()
                logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors['token_type_ids'], attention_mask=merged_tensors['attention_mask'])

                logits_lbls = logits.logits[:num_lb]
                logits_ulbl = logits.logits[num_lb:]


                # ------ mixup loss here ---------
                labels_lbls = F.one_hot(cuda_tensors_supervised['lbl'],num_classes=logits_lbls.shape[1])
                labels_ulbl = F.one_hot(cuda_tensors_unsupervised['lbl'],num_classes=logits_ulbl.shape[1])
                
                lam = np.random.beta(alpha,alpha)
                lam = max(lam, 1 - lam)
                W_logits_ = logits.logits
                W_labels_ = torch.cat((labels_lbls, labels_ulbl))
                shuffled_ind = torch.randperm(num_lb+num_ulb)
                W_logits = W_logits_[shuffled_ind]
                W_labels = W_labels_[shuffled_ind]
                # print("W_labels", W_labels)
        
                X_logits = logits_lbls * lam + W_logits[:num_lb] * (1-lam)
                X_labels = labels_lbls * lam + W_labels[:num_lb] * (1-lam)

                U_logits = logits_ulbl * lam + W_logits[num_lb:] * (1-lam)
                U_labels = labels_ulbl * lam + W_labels[num_lb:] * (1-lam)

                X_loss = loss_fn_supervised(X_logits, X_labels)
                U_loss = torch.mean(torch.sum(-U_labels * torch.log_softmax(U_logits, dim=-1), dim=0))

                loss = 0.5 * X_loss + 0.5 * U_loss

                loss.backward()
                optimizer.step()

            f1_macro_validation, loss_validation, ece = evaluate(
                model, n_classes, validation_dataloader, loss_fn, unsup_batch_size)
            print('Confident learning metrics', f1_macro_validation)

            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation
                if best_f1 > best_f1_overall:
                    #model.save_pretrained(model_dir+"/ust")
                    if not os.path.exists("data/" + model_dir):
                        os.makedirs("data/" + model_dir)
                    torch.save(model.state_dict(), "data/" +model_dir + "/pytorch_model.bin")
                    best_f1_overall = best_f1
                print('New best macro validation', best_f1, 'Epoch', epoch)
                continue
        
            if crt_patience == 3:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

            crt_patience += 1


    # load the best model and evaluate on test set
    copy_cfg.return_dict = True
    model = BertModel(pt_teacher_checkpoint, num_labels=n_classes)
    model.to(device)
    state_dict = torch.load("data/" +model_dir + "/pytorch_model.bin")
    model.load_state_dict(state_dict)

    f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size)
    logger.info ("Test macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best mixmatch model"] = {}
    logger_dict["Best mixmatch model"]["F1 before temp scaling"] = str(f1_macro_test)
    logger_dict["Best mixmatch model"]["ECE before temp scaling"] = str(ece_metric)


    logger_dict["Best mixmatch model"]["T before temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(validation_dataloader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = model(cuda_tensors['input_ids'], cuda_tensors['token_type_ids'], cuda_tensors['attention_mask'], True)

                loss = loss_fn(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()


        f1_macro_test, loss_test, ece_metric = evaluate(model, n_classes, test_dataloader, loss_fn, unsup_batch_size, True)

        logger_dict["Best mixmatch model"]["F1 after temp scaling"] = str(f1_macro_test)
        logger_dict["Best mixmatch model"]["ECE after temp scaling"] = str(ece_metric)

        logger_dict["Best mixmatch model"]["T  after temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    with open("data/" + model_dir +"/"+ results_file + '.txt','w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))

    # Log Final Test Metrics to WandB
    if wandb.run:
        wandb.log({
            "test_macro-F1": f1_macro_test,
             "test_ece": ece_metric.item() if isinstance(ece_metric, torch.Tensor) else ece_metric,
             "test_loss": loss_test
        })





