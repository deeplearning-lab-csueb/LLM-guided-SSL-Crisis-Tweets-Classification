import argparse
from aum import AUMCalculator

from collections import defaultdict
from copy import deepcopy

import json
import logging
import matplotlib.pyplot as plt

import numpy as np

import os
import pandas as pd
import wandb

from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# logging
logger = logging.getLogger('UST')
logging.basicConfig(level=logging.INFO)

GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED", "42"))
logger.info("Global seed {}".format(GLOBAL_SEED))



# ── GPU setup ──────────────────────────────────────────────────────────────────
if torch.cuda.device_count() >= 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is:", device)


def multigpu(model):
    if torch.cuda.device_count() >= 2:
        return nn.DataParallel(model).to(device)
    return model.to(device)


# ── Model ──────────────────────────────────────────────────────────────────────
class BertModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels, token=None):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, token=token)
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, temperature_scaling=False):
        if temperature_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits = outputs.logits / temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return outputs


def get_temperature_param(model):
    if isinstance(model, nn.DataParallel):
        return model.module.T
    return model.T


def get_temperature_value(model):
    return float(get_temperature_param(model).detach().cpu().item())


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, test_dataloader, criterion, batch_size, num_labels, temp_scaling=False):
    full_predictions, true_labels, probabilities = [], [], []
    was_training = model.training
    model.eval()
    crt_loss = 0.0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {k: elem[k].to(device) for k in elem if k not in ['idx', 'weights']}
            logits = model(input_ids=x['input_ids'],
                           token_type_ids=x.get('token_type_ids'),
                           attention_mask=x['attention_mask'],
                           temperature_scaling=temp_scaling)
            results = torch.argmax(logits.logits, dim=1)
            prob = F.softmax(logits.logits, dim=1).cpu()
            probabilities.append(prob)
            crt_loss += criterion(logits.logits, x['lbl']).cpu().item()
            full_predictions.append(results.cpu())
            true_labels.append(elem['lbl'].cpu())

    if was_training:
        model.train()

    preds = torch.cat(probabilities).to(device)
    full_predictions = torch.cat(full_predictions).numpy()
    true_labels = torch.cat(true_labels).numpy()
    orig = torch.tensor(true_labels, dtype=torch.long, device=device)
    metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm='l1').to(device)
    ece_metric = metric(preds, orig)

    return (f1_score(true_labels, full_predictions, average='macro'),
            crt_loss / len(test_dataloader),
            ece_metric)


# ── Pseudo-labelling ───────────────────────────────────────────────────────────
def predict_unlabeled(model, ds_unlabeled):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        ds_unlabeled, batch_size=128, shuffle=False)
    y_pred_unlbl = []

    with torch.no_grad():
        for elem in data_loader:
            x = {k: elem[k].to(device) for k in elem if k not in ['idx', 'weights']}
            pred = model(input_ids=x['input_ids'],
                         token_type_ids=x.get('token_type_ids'),
                         attention_mask=x['attention_mask'])
            y_pred_unlbl.append(pred.logits.cpu())

    y_pred_unlbl = torch.cat(y_pred_unlbl, dim=0).numpy()
    y_pred_unlbl = np.argmax(y_pred_unlbl, axis=-1).flatten()
    return CustomDataset(
        ds_unlabeled.text_list, y_pred_unlbl, ds_unlabeled.idxes,
        ds_unlabeled.tokenizer, labeled=True, encodings=ds_unlabeled.encodings)


# ── SSL with AUM tracking ──────────────────────────────────────────────────────
def train_ssl_with_aum(pt_teacher_checkpoint, ds_train, ds_pseudolabeled, token=None,
                       ulb_epochs=20, aum_calculator=None, ls=0.0, sup_batch_size=16, num_labels=10):
    model = multigpu(BertModel(pt_teacher_checkpoint, num_labels=num_labels, token=token))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

    data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10 ** 4)
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
    data_loader_unlabeled = torch.utils.data.DataLoader(
        ds_pseudolabeled, batch_size=128, shuffle=False)

    for epoch in range(ulb_epochs):
        for data_supervised, data_unsupervised in tqdm(
                zip(train_dataloader, data_loader_unlabeled)):
            cuda_sup = {k: data_supervised[k].to(device)
                        for k in data_supervised if k not in ['idx', 'weights']}
            cuda_unsup = {k: data_unsupervised[k].to(device)
                          for k in data_unsupervised if k not in ['idx', 'weights']}

            optimizer.zero_grad()
            logits_lbls = model(input_ids=cuda_sup['input_ids'],
                                token_type_ids=cuda_sup.get('token_type_ids'),
                                attention_mask=cuda_sup['attention_mask']).logits
            logits_ulbl = model(input_ids=cuda_unsup['input_ids'],
                                token_type_ids=cuda_unsup.get('token_type_ids'),
                                attention_mask=cuda_unsup['attention_mask']).logits

            aum_calculator.update(logits_ulbl.detach(), cuda_unsup['lbl'],
                                  data_unsupervised['idx'].numpy())

            loss_sup = loss_fn_supervised(logits_lbls, cuda_sup['lbl'])
            loss_unsup = loss_fn_unsupervised(logits_ulbl, cuda_unsup['lbl'])
            loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
            loss.backward()
            optimizer.step()


# ── SSL with Mixup (no AUM) ────────────────────────────────────────────────────
def train_ssl_no_aum_with_mixup(pt_teacher_checkpoint, ds_train, val_dataloader, token=None,
                                ds_low_aum=None, ds_high_aum=None, ulb_epochs=20, ls=0.0, model_dir="",
                                sup_batch_size=16, best_f1_overall=0, best_f1=0, num_labels=10):
    save_path = f"data/{model_dir}/pytorch_model.bin"
    model = multigpu(BertModel(pt_teacher_checkpoint, num_labels, token=token))
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

    data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10 ** 4)
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
    unlabeled_low = torch.utils.data.DataLoader(ds_low_aum, batch_size=128, shuffle=False)
    unlabeled_high = torch.utils.data.DataLoader(ds_high_aum, batch_size=128, shuffle=False)

    crt_patience = 0
    os.makedirs(f"data/{model_dir}", exist_ok=True)
    save_path = f"data/{model_dir}/pytorch_model.bin"
    for epoch in range(ulb_epochs):
        for data_supervised, data_unsup_low, data_unsup_high in tqdm(
                zip(train_dataloader, unlabeled_low, unlabeled_high)):
            cuda_sup = {k: data_supervised[k].to(device)
                        for k in data_supervised if k not in ['idx', 'weights']}
            cuda_low = {k: data_unsup_low[k].to(device)
                        for k in data_unsup_low if k not in ['idx', 'weights']}
            cuda_high = {k: data_unsup_high[k].to(device)
                         for k in data_unsup_high if k not in ['idx', 'weights']}

            num_lb = cuda_sup['input_ids'].shape[0]
            num_ulb_low = cuda_low['input_ids'].shape[0]
            num_ulb_high = cuda_high['input_ids'].shape[0]  # BUG FIX: was num_ulb_hight

            # Forward pass on all three splits at once
            merged = {}
            for k in cuda_sup:
                merged[k] = torch.cat((cuda_sup[k], cuda_low[k], cuda_high[k]))

            optimizer.zero_grad()
            logits = model(input_ids=merged['input_ids'],
                           token_type_ids=merged.get('token_type_ids'),
                           attention_mask=merged['attention_mask'])

            logits_lbls = logits.logits[:num_lb]
            logits_ulbl_low = logits.logits[num_lb:num_lb + num_ulb_low]
            logits_ulbl_high = logits.logits[num_lb + num_ulb_low:]

            alpha_mix = 0.4
            lam = np.random.beta(alpha_mix, alpha_mix)

            labels_lbls = F.one_hot(cuda_sup['lbl'], num_classes=logits_lbls.shape[1]).float()
            labels_ulbl_low = F.one_hot(cuda_low['lbl'], num_classes=logits_ulbl_low.shape[1]).float()
            labels_ulbl_high = F.one_hot(cuda_high['lbl'], num_classes=logits_ulbl_high.shape[1]).float()

            batch_size = logits_lbls.shape[0]
            mix_size_1 = min(batch_size, logits_ulbl_low.shape[0])
            mix_size_2 = min(logits_ulbl_high.shape[0], logits_ulbl_low.shape[0])

            # Mixup: labeled ↔ low-AUM unlabeled
            M_logits_1 = lam * logits_lbls[:mix_size_1] + (1 - lam) * logits_ulbl_low[:mix_size_1]
            M_labels_1 = lam * labels_lbls[:mix_size_1] + (1 - lam) * labels_ulbl_low[:mix_size_1]

            # Mixup: high-AUM ↔ low-AUM unlabeled
            M_logits_2 = lam * logits_ulbl_high[:mix_size_2] + (1 - lam) * logits_ulbl_low[:mix_size_2]
            M_labels_2 = lam * labels_ulbl_high[:mix_size_2] + (1 - lam) * labels_ulbl_low[:mix_size_2]

            loss_lbl = loss_fn_supervised(logits_lbls, cuda_sup['lbl'])
            loss_ulbl_high = loss_fn_unsupervised(logits_ulbl_high, cuda_high['lbl'])
            loss_M1 = torch.mean(
                torch.sum(-M_labels_1 * torch.log_softmax(M_logits_1, dim=-1), dim=-1))
            loss_M2 = torch.mean(
                torch.sum(-M_labels_2 * torch.log_softmax(M_logits_2, dim=-1), dim=-1))

            loss = 0.25 * loss_lbl + 0.25 * torch.mean(loss_ulbl_high) + \
                   0.25 * loss_M1 + 0.25 * loss_M2
            loss.backward()
            optimizer.step()

        # BUG FIX: evaluate once per epoch, not once per batch
        f1_val, loss_val, ece = evaluate(model, val_dataloader, loss_fn_supervised, 128, num_labels)
        print(f'Validation F1: {f1_val:.4f}  (epoch {epoch})')
        if wandb.run:
            wandb.log({
                "unsup_train_loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                "dev_macro-F1": f1_val,
                "dev_ece": ece.item() if isinstance(ece, torch.Tensor) else ece,
                "unsup_epoch": epoch
            })

        if f1_val >= best_f1:
            crt_patience = 0
            best_f1 = f1_val
            if best_f1 > best_f1_overall:
                torch.save(model.state_dict(), save_path)
                best_f1_overall = best_f1
            print(f'New best macro validation {best_f1:.4f}  epoch {epoch}')
            continue

        if crt_patience == 3:
            crt_patience = 0
            print('Exceeding max patience; Exiting..')
            break

        crt_patience += 1

    return best_f1_overall, best_f1


# ── Main self-training loop (AUM + Mixup) ─────────────────────────────────────
def train_model_st_with_aummixup(ds_train, ds_dev, ds_test, ds_unlabeled,
                                  pt_teacher_checkpoint, cfg, model_dir,
                                  aum_save_dir, num_labels,
                                  sup_batch_size=16, unsup_batch_size=64,
                                  unsup_size=4096, sample_size=16384,
                                  sample_scheme='easy_bald_class_conf',
                                  T=30, alpha=0.1, sup_epochs=20,
                                  unsup_epochs=25, N_base=10,
                                  dense_dropout=0.5,
                                  attention_probs_dropout_prob=0.3,
                                  hidden_dropout_prob=0.3,
                                  results_file="", temp_scaling=False, ls=0.0,
                                  run_name="aum_mixup_run", token=None):

    import uuid
    aum_save_dir = os.path.join(aum_save_dir, f"{run_name}_{uuid.uuid4().hex[:8]}")
    os.makedirs(aum_save_dir, exist_ok=True)

    logger_dict = {"Temperature Scaling": temp_scaling, "Label Smoothing": ls}

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=sup_batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(
        ds_dev, batch_size=sup_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=128, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls)
    os.makedirs(f"data/{model_dir}", exist_ok=True)
    save_path = f"data/{model_dir}/pytorch_model.bin"

    best_f1_overall = 0
    best_f1 = 0  #BUG FIX: initialise here so it is always defined
    crt_patience = 0

    # ── Phase 1: supervised warm-up ──
    for counter in range(N_base):
        best_f1 = 0
        model = multigpu(BertModel(pt_teacher_checkpoint, num_labels))
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        if counter == 0:
            logger.info(model)

        for epoch in range(sup_epochs):
            for data in tqdm(train_dataloader):
                cuda = {k: data[k].to(device)
                        for k in data if k not in ['idx', 'weights']}
                optimizer.zero_grad()
                logits = model(input_ids=cuda['input_ids'],
                               token_type_ids=cuda.get('token_type_ids'),
                               attention_mask=cuda['attention_mask'])
                loss = loss_fn(logits.logits, cuda['lbl'])
                loss.backward()
                optimizer.step()

            f1_val, _, _ = evaluate(model, validation_dataloader, loss_fn,
                                    unsup_batch_size, num_labels)
            if f1_val >= best_f1:
                crt_patience = 0
                best_f1 = f1_val
                if best_f1 > best_f1_overall:
                    torch.save(model.state_dict(), save_path)
                    best_f1_overall = best_f1
                print(f'New best macro validation {best_f1:.4f}  epoch {epoch}')
                continue

            if crt_patience == 3:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

            crt_patience += 1

    del model

    # ── Phase 2: self-training with AUM-guided Mixup ──
    best_model = multigpu(BertModel(pt_teacher_checkpoint, num_labels))
    best_model.load_state_dict(torch.load(save_path))

    for epoch in range(unsup_epochs):
        # Reload teacher from best checkpoint to use improved pseudo-labels
        best_model.load_state_dict(torch.load(save_path))
        aum_calculator = AUMCalculator(aum_save_dir, compressed=False)
        pseudolabeled_data = predict_unlabeled(best_model, ds_unlabeled)
        train_ssl_with_aum(pt_teacher_checkpoint, ds_train, pseudolabeled_data, token=token,
                           ulb_epochs=sup_epochs, aum_calculator=aum_calculator, ls=ls, sup_batch_size=sup_batch_size, num_labels=num_labels)
        aum_calculator.finalize()

        aum_values_df = pd.read_csv(os.path.join(aum_save_dir, 'aum_values.csv'))
        aum_values = sorted(aum_values_df['aum'].tolist())
        median_aum_value = aum_values[int(len(aum_values) * 0.5)]

        high_aum_ids, low_aum_ids = [], []
        for _, row in aum_values_df.iterrows():
            (high_aum_ids if row['aum'] > median_aum_value else low_aum_ids).append(
                int(row['sample_id']))

        print(f"Low AUM: {len(low_aum_ids)}  High AUM: {len(high_aum_ids)}")

        low_aum_data = pseudolabeled_data.get_subset_dataset(low_aum_ids)
        high_aum_data = pseudolabeled_data.get_subset_dataset(high_aum_ids)

        ds_unlabeled_low = CustomDataset(
            low_aum_data.text_list, low_aum_data.labels, low_aum_data.idxes, ds_unlabeled.tokenizer, labeled=True, encodings=low_aum_data.encodings)
        ds_unlabeled_high = CustomDataset(
            high_aum_data.text_list, high_aum_data.labels, high_aum_data.idxes, ds_unlabeled.tokenizer, labeled=True, encodings=high_aum_data.encodings)

        best_f1_overall, best_f1 = train_ssl_no_aum_with_mixup(
            pt_teacher_checkpoint, ds_train, validation_dataloader, token=token,
            ds_low_aum=ds_unlabeled_low, ds_high_aum=ds_unlabeled_high, ulb_epochs=sup_epochs, ls=ls, model_dir=model_dir,
            sup_batch_size=sup_batch_size, best_f1_overall=best_f1_overall, best_f1=best_f1, num_labels=num_labels)

    # ── Final evaluation ──
    # BUG FIX: load best checkpoint into a fresh model for evaluation;
    # previously the code evaluated `best_model` (the AUM-phase seed model)
    # instead of the final best model saved during mixup training.
    final_model = multigpu(BertModel(pt_teacher_checkpoint, num_labels, token=token))
    final_model.load_state_dict(torch.load(save_path))
    final_model.eval()

    rel_file = f"{model_dir}_{results_file}"
    f1_macro_test, loss_test, ece_metric = evaluate(final_model, test_dataloader, loss_fn,
                                                    unsup_batch_size, num_labels)
    logger.info(f"Test macro F1 based on best validation f1: {f1_macro_test}")

    logger_dict["Best ST+AumMixup model"] = {
        "F1 before temp scaling": str(f1_macro_test),
        "ECE before temp scaling": str(ece_metric),
        "T before temp scaling": str(get_temperature_value(final_model)),
    }

    if temp_scaling:
        calibration_loss_fn = torch.nn.CrossEntropyLoss()
        temp_param = get_temperature_param(final_model)
        with torch.no_grad():
            temp_param.fill_(1.0)
            temp_param.clamp_(min=1e-3, max=100.0)

        final_model.eval()
        optimizer = torch.optim.Adam([temp_param], lr=2e-2)
        for ep in range(20):
            for data in tqdm(validation_dataloader):
                cuda = {k: data[k].to(device)
                        for k in data if k not in ['idx', 'weights']}
                optimizer.zero_grad()
                result = final_model(cuda['input_ids'], cuda.get('token_type_ids'),
                                     cuda['attention_mask'], True)
                loss = calibration_loss_fn(result.logits, cuda['lbl'])
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    temp_param.clamp_(min=1e-3, max=100.0)

        f1_macro_test, _, ece_metric = evaluate(
            final_model, test_dataloader, loss_fn, unsup_batch_size, num_labels, True)
        logger_dict["Best ST+AumMixup model"]["F1 after temp scaling"] = str(f1_macro_test)
        logger_dict["Best ST+AumMixup model"]["ECE after temp scaling"] = str(ece_metric)
        logger_dict["Best ST+AumMixup model"]["T after temp scaling"] = str(get_temperature_value(final_model))

    f1_macro_test_final = float(logger_dict["Best ST+AumMixup model"].get("F1 after temp scaling", logger_dict["Best ST+AumMixup model"].get("F1 before temp scaling")))
    if wandb.run:
        wandb.log({
            "test_macro-F1": f1_macro_test_final,
            "test_ece": ece_metric.item() if isinstance(ece_metric, torch.Tensor) else ece_metric
        })

    print(json.dumps(logger_dict, indent=4))
    out_path = f"data/{model_dir}/{results_file}.txt"
    with open(out_path, 'w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))

    logger.info("Generating predictions for artifact...")
    final_model.eval()
    all_preds = []
    all_golds = []
    
    final_test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=128, shuffle=False)
    with torch.no_grad():
        for elem in tqdm(final_test_dataloader):
            x = {k: elem[k].to(device) for k in elem if k not in ['idx', 'weights']}
            logits = final_model(input_ids=x['input_ids'], token_type_ids=x.get('token_type_ids'), attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            preds = torch.argmax(logits.logits, dim=1).cpu().detach().numpy()
            golds = elem['lbl'].cpu().detach().numpy()
            all_preds.extend(preds)
            all_golds.extend(golds)
            
    all_ids = ds_test.idxes
    
    if wandb.run:
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
        artifact.add_file(pred_file, name=f"{run_name}.csv")
        wandb.log_artifact(artifact)
        os.remove(pred_file)
        
    try:
        import shutil
        if os.path.exists(save_path): os.remove(save_path)
        if os.path.exists(out_path): os.remove(out_path)
        if os.path.exists(aum_save_dir): shutil.rmtree(aum_save_dir)
        logger.info("Cleaned up local model, results files, and AUM artifacts.")
    except Exception as e:
        logger.warning(f"Failed to clean up local files: {e}")


# ── SSL with Saliency-guided Mixup (no AUM) ───────────────────────────────────
def train_ssl_no_aum_with_sal_mixup(pt_teacher_checkpoint, ds_train, val_dataloader, token=None,
                                    ds_low_aum=None, ds_high_aum=None, ulb_epochs=20, ls=0.0, model_dir="",
                                    sup_batch_size=16, best_f1_overall=0, best_f1=0, num_labels=10):
    save_path = f"data/{model_dir}/pytorch_model.bin"
    model = multigpu(BertModel(pt_teacher_checkpoint, num_labels=num_labels, token=token))
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    data_sampler = torch.utils.data.RandomSampler(ds_train, num_samples=10 ** 5, replacement=True)
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_sampler=batch_sampler)
    unlabeled_low = torch.utils.data.DataLoader(ds_low_aum, batch_size=128, shuffle=False)
    unlabeled_high = torch.utils.data.DataLoader(ds_high_aum, batch_size=128, shuffle=False)

    crt_patience = 0
    os.makedirs(f"data/{model_dir}", exist_ok=True)
    save_path = f"data/{model_dir}/pytorch_model.bin"

    for epoch in range(ulb_epochs):
        for data_supervised, data_unsup_low, data_unsup_high in tqdm(
                zip(train_dataloader, unlabeled_low, unlabeled_high)):
            cuda_sup = {k: data_supervised[k].to(device)
                        for k in data_supervised if k not in ['idx', 'weights']}
            cuda_low = {k: data_unsup_low[k].to(device)
                        for k in data_unsup_low if k not in ['idx', 'weights']}
            cuda_high = {k: data_unsup_high[k].to(device)
                         for k in data_unsup_high if k not in ['idx', 'weights']}

            num_lb = cuda_sup['input_ids'].shape[0]
            num_ulb_low = cuda_low['input_ids'].shape[0]
            num_ulb_high = cuda_high['input_ids'].shape[0]

            merged = {}
            for k in cuda_sup:
                merged[k] = torch.cat((cuda_sup[k], cuda_low[k], cuda_high[k]))

            optimizer.zero_grad()
            logits = model(input_ids=merged['input_ids'],
                           token_type_ids=merged.get('token_type_ids'),
                           attention_mask=merged['attention_mask'])

            logits_lbls = logits.logits[:num_lb]
            logits_ulbl_low = logits.logits[num_lb:num_lb + num_ulb_low]
            logits_ulbl_high = logits.logits[num_lb + num_ulb_low:]

            # BUG FIX: use a single backward pass for saliency gradients.
            # Previously three separate .backward(retain_graph=True) calls ran
            # before the final loss.backward(), causing gradients to accumulate
            # and the graph to be traversed multiple times incorrectly.
            loss_for_grad = (loss_fn_supervised(logits_lbls, cuda_sup['lbl']) +
                             (cuda_high['weights'] * loss_fn_unsupervised(
                                 logits_ulbl_high, cuda_high['lbl'])).mean() +
                             (cuda_low['weights'] * loss_fn_unsupervised(
                                 logits_ulbl_low, cuda_low['lbl'])).mean())

            logits_lbls.retain_grad()
            logits_ulbl_low.retain_grad()
            logits_ulbl_high.retain_grad()
            loss_for_grad.backward(retain_graph=True)

            lbl_grads = logits_lbls.grad.data.abs()
            ulbl_high_grads = logits_ulbl_high.grad.data.abs()
            ulbl_low_grads = logits_ulbl_low.grad.data.abs()

            alpha_mix = 0.4
            lam = np.random.beta(alpha_mix, alpha_mix)

            labels_lbls = F.one_hot(cuda_sup['lbl'], num_classes=logits_lbls.shape[1]).float()
            labels_ulbl_low = F.one_hot(cuda_low['lbl'], num_classes=logits_ulbl_low.shape[1]).float()
            labels_ulbl_high = F.one_hot(cuda_high['lbl'], num_classes=logits_ulbl_high.shape[1]).float()

            batch_size = logits_lbls.shape[0]
            mix_size_1 = min(batch_size, logits_ulbl_low.shape[0])
            mix_size_2 = min(logits_ulbl_high.shape[0], logits_ulbl_low.shape[0])

            # ── Saliency-guided mixup for labeled ↔ low-AUM ──
            sim_matrix = F.cosine_similarity(lbl_grads[:mix_size_1].unsqueeze(1), ulbl_low_grads[:mix_size_1].unsqueeze(0), dim=2)
            argmax_idx = torch.argmax(sim_matrix, dim=1)
            argmin_idx = torch.argmin(sim_matrix, dim=1)

            similar = logits_ulbl_low[:mix_size_1][argmax_idx]
            similar_label = labels_ulbl_low[:mix_size_1][argmax_idx]
            dissimilar = logits_ulbl_low[:mix_size_1][argmin_idx]
            dissimilar_label = labels_ulbl_low[:mix_size_1][argmin_idx]

            M_logits_1_ood = lam * logits_lbls[:mix_size_1] + (1 - lam) * dissimilar
            M_labels_1_ood = lam * labels_lbls[:mix_size_1] + (1 - lam) * dissimilar_label
            M_logits_1_id = lam * logits_lbls[:mix_size_1] + (1 - lam) * similar
            M_labels_1_id = lam * labels_lbls[:mix_size_1] + (1 - lam) * similar_label

            # ── Saliency-guided mixup for high-AUM ↔ low-AUM ──
            sim_matrix_h = F.cosine_similarity(ulbl_high_grads[:mix_size_2].unsqueeze(1), ulbl_low_grads[:mix_size_2].unsqueeze(0), dim=2)
            argmax_idx_h = torch.argmax(sim_matrix_h, dim=1)
            argmin_idx_h = torch.argmin(sim_matrix_h, dim=1)

            similar_h = logits_ulbl_low[:mix_size_2][argmax_idx_h]
            similar_label_h = labels_ulbl_low[:mix_size_2][argmax_idx_h]
            dissimilar_h = logits_ulbl_low[:mix_size_2][argmin_idx_h]
            dissimilar_label_h = labels_ulbl_low[:mix_size_2][argmin_idx_h]

            M_logits_2_ood = lam * logits_ulbl_high[:mix_size_2] + (1 - lam) * dissimilar_h
            M_labels_2_ood = lam * labels_ulbl_high[:mix_size_2] + (1 - lam) * dissimilar_label_h
            M_logits_2_id = lam * logits_ulbl_high[:mix_size_2] + (1 - lam) * similar_h
            M_labels_2_id = lam * labels_ulbl_high[:mix_size_2] + (1 - lam) * similar_label_h

            loss_lbl = loss_fn_supervised(logits_lbls, cuda_sup['lbl'])
            loss_ulbl_high = loss_fn_unsupervised(logits_ulbl_high, cuda_high['lbl'])
            loss_M1 = (torch.mean(torch.sum(-M_labels_1_ood * torch.log_softmax(M_logits_1_ood, dim=-1), dim=-1)) +
                       torch.mean(torch.sum(-M_labels_1_id * torch.log_softmax(M_logits_1_id, dim=-1), dim=-1))) / 2
            loss_M2 = (torch.mean(torch.sum(-M_labels_2_ood * torch.log_softmax(M_logits_2_ood, dim=-1), dim=-1)) +
                       torch.mean(torch.sum(-M_labels_2_id * torch.log_softmax(M_logits_2_id, dim=-1), dim=-1))) / 2

            loss = 0.25 * loss_lbl + 0.25 * torch.mean(loss_ulbl_high) + \
                   0.25 * loss_M1 + 0.25 * loss_M2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # BUG FIX: evaluate once per epoch, not once per batch
        f1_val, _, ece_metric = evaluate(model, val_dataloader, loss_fn_supervised, 128, num_labels)
        print(f'Validation F1: {f1_val:.4f}  (epoch {epoch})')

        if f1_val >= best_f1:
            crt_patience = 0
            best_f1 = f1_val
            if best_f1 > best_f1_overall:
                torch.save(model.state_dict(), save_path)
                best_f1_overall = best_f1
            print(f'New best macro validation {best_f1:.4f}  epoch {epoch}')
            continue

        if crt_patience == 3:
            crt_patience = 0
            print('Exceeding max patience; Exiting..')
            break

        crt_patience += 1

    return best_f1_overall, best_f1


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, idxes, tokenizer, max_seq_len=128, labeled=True, encodings=None):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labeled = labeled
        self.idxes = idxes
        self.weights = [1] * len(self.text_list)
        
        if encodings is None:
            tok = self.tokenizer(
                self.text_list, padding='max_length', max_length=self.max_seq_len, truncation=True)
            self.encodings = {key: torch.tensor(tok[key]) for key in tok}
        else:
            self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labeled:
            item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['idx'] = self.idxes[idx]
        item['weights'] = self.weights[idx]
        return item

    def __len__(self):
        return len(self.text_list)

    def get_subset_dataset(self, idxs):
        # BUG FIX: was O(n²) nested loop; now O(n) using a dict mapping id -> position
        idxs_set = set(idxs)
        text_lists, label_lists, id_lists = [], [], []
        indices = []
        for pos, (text, label, id_) in enumerate(zip(self.text_list, self.labels, self.idxes)):
            if id_ in idxs_set:
                text_lists.append(text)
                label_lists.append(label)
                id_lists.append(id_)
                indices.append(pos)
        
        new_encodings = {key: val[indices] for key, val in self.encodings.items()}
        return CustomDataset(text_lists, label_lists, id_lists, self.tokenizer, self.max_seq_len, self.labeled, encodings=new_encodings)


def get_label_to_id(args):
    if args["dataset"] == 'humanitarian8':
        return {
            "caution_and_advice": 0,
            "displaced_people_and_evacuations": 1,
            "infrastructure_and_utility_damage": 2,
            "not_humanitarian": 3,
            "other_relevant_information": 4,
            "requests_or_urgent_needs": 5,
            "rescue_volunteering_or_donation_effort": 6,
            "sympathy_and_support": 7,
        }
    elif args["dataset"] == 'humanitarian9':
        return {
            "caution_and_advice": 0,
            "displaced_people_and_evacuations": 1,
            "infrastructure_and_utility_damage": 2,
            "injured_or_dead_people": 3,
            "not_humanitarian": 4,
            "other_relevant_information": 5,
            "requests_or_urgent_needs": 6,
            "rescue_volunteering_or_donation_effort": 7,
            "sympathy_and_support": 8,
        }
    elif args["dataset"] == 'humanitarian10':
        return {
            "caution_and_advice": 0,
            "displaced_people_and_evacuations": 1,
            "infrastructure_and_utility_damage": 2,
            "injured_or_dead_people": 3,
            "missing_or_found_people": 4,
            "not_humanitarian": 5,
            "other_relevant_information": 6,
            "requests_or_urgent_needs": 7,
            "rescue_volunteering_or_donation_effort": 8,
            "sympathy_and_support": 9,
        }
    else:
        raise ValueError(f"Unknown dataset: {args['dataset']}")


def get_dataset(path, tokenizer, label_to_id, labeled=True):
    df = pd.read_csv(path, sep='\t')
    text_list, labels_list, ids_list = [], [], []
    for _, row in df.iterrows():
        if pd.isna(row['tweet_text']):
            continue
        text_list.append(row['tweet_text'])
        labels_list.append(label_to_id[row['class_label']])
        ids_list.append(row['tweet_id'])
    return CustomDataset(text_list, labels_list, ids_list, tokenizer, labeled=labeled)


# Runner script handles the arguments now.
