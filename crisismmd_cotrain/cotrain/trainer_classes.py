import os
import time
import copy
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

#local imports
from utils import *


class WeightGenerator:
    def __init__(self, args, dataloaders: Dict, training_params: Dict, optimizer_params: Dict,
                 hyper_params: Dict, devices: Tuple[str, str], models: Tuple[torch.nn.Module, torch.nn.Module],
                 auto_labeled_data: pd.DataFrame):
        self.args = args
        self.dataloaders = dataloaders
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.hyper_params = hyper_params
        self.device_1, self.device_2 = devices
        self.model_1, self.model_2 = models
        self.auto_labeled_data = auto_labeled_data
        self.metric_combination = args.metric_combination
        
        self._initialize_components()
        self._setup_tracking()
        
    def _initialize_components(self):
        """Initialize models and optimizers."""
        self.model_1.to(self.device_1)
        self.model_2.to(self.device_2)
        
        self.optimizer_1 = self.optimizer_params['optimizer_1']
        self.optimizer_2 = self.optimizer_params['optimizer_2']
        self.lr_scheduler_1 = self.optimizer_params['lr_scheduler_1']
        self.lr_scheduler_2 = self.optimizer_params['lr_scheduler_2']
        self.criterion = self.optimizer_params['criterion']
        
    def _setup_tracking(self):
        """Initialize data structures for tracking probabilities."""
        self.id2index_auto_label = self._create_id_to_index_mapping(self.dataloaders['auto_label_dataloader'])
        self.num_samples = len(self.id2index_auto_label)
        
        # Preallocate storage for probabilities
        self.all_epoch_probabilities_1 = np.zeros((self.num_samples, self.training_params['num_epochs']), dtype=np.float32)
        self.all_epoch_probabilities_2 = np.zeros((self.num_samples, self.training_params['num_epochs']), dtype=np.float32)
        self.best_f1 = -1
        self.best_epoch = 0
        
    @staticmethod
    def _create_id_to_index_mapping(dataloader) -> Dict:
        """Create mapping from ID to index."""
        id_to_index = {}
        global_index = 0
        
        for batch in dataloader:
            ids = batch['id']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            elif not isinstance(ids, list):
                ids = [ids]
                
            for id_value in ids:
                id_to_index[id_value] = global_index
                global_index += 1
                
        return id_to_index
        
    def _train_models(self, batch_1, batch_2):
        """Train both models on a batch of data."""
        # Model 1 training
        batch_1 = {k: v.to(self.device_1) for k, v in batch_1.items()}
        outputs_1 = self.model_1(input_ids=batch_1['input_ids'], attention_mask=batch_1['attention_mask'])
        loss_1 = torch.mean(self.criterion(outputs_1, batch_1['labels'])) / self.training_params['accumulation_steps']
        loss_1.backward()

        # Model 2 training
        batch_2 = {k: v.to(self.device_2) for k, v in batch_2.items()}
        outputs_2 = self.model_2(input_ids=batch_2['input_ids'], attention_mask=batch_2['attention_mask'])
        loss_2 = torch.mean(self.criterion(outputs_2, batch_2['labels'])) / self.training_params['accumulation_steps']
        loss_2.backward()
        
        return outputs_1, outputs_2
        
    def _update_models(self, i: int, total_batches: int):
        """Update models based on accumulation steps."""
        if (i + 1) % self.training_params['accumulation_steps'] == 0 or (i + 1) == total_batches:
            torch.nn.utils.clip_grad_norm_(self.model_1.parameters(), self.training_params.get('max_grad_norm', 1.0))
            self.optimizer_1.step()
            self.lr_scheduler_1.step()
            self.optimizer_1.zero_grad()

            torch.nn.utils.clip_grad_norm_(self.model_2.parameters(), self.training_params.get('max_grad_norm', 1.0))
            self.optimizer_2.step()
            self.lr_scheduler_2.step()
            self.optimizer_2.zero_grad()
            
    def _parallel_inference(self, batch):
        """Run model inference. Use sequential execution if on same device to save memory."""
        input_ids = batch['input_ids'].to(self.device_1)
        attention_mask = batch['attention_mask'].to(self.device_1)
        labels = batch['labels'].to(self.device_1)
        ids = batch['id']
        
        outputs_1, outputs_2 = None, None

        # Check if we are on the same device (e.g. single GPU)
        # If so, run sequentially to avoid OOM (double memory usage)
        if self.device_1 == self.device_2:
            try:
                outputs_1 = self.model_1(input_ids=input_ids, attention_mask=attention_mask)
                outputs_2 = self.model_2(
                    input_ids=input_ids.to(self.device_2),
                    attention_mask=attention_mask.to(self.device_2)
                )
            except RuntimeError as e:
                # If OOM happens even sequentially, re-raise with context
                if "out of memory" in str(e):
                    print(f"OOM during sequential inference on {self.device_1}")
                raise e
        else:
            # Different devices (multi-GPU), run in parallel threads
            def run_model_1():
                nonlocal outputs_1
                try:
                    outputs_1 = self.model_1(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"Error in model 1 thread: {e}")

            def run_model_2():
                nonlocal outputs_2
                try:
                    outputs_2 = self.model_2(
                        input_ids=input_ids.to(self.device_2),
                        attention_mask=attention_mask.to(self.device_2)
                    )
                except Exception as e:
                    print(f"Error in model 2 thread: {e}")

            t1 = threading.Thread(target=run_model_1)
            t2 = threading.Thread(target=run_model_2)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        
        if outputs_1 is None or outputs_2 is None:
            raise RuntimeError("Model inference failed (outputs are None). potentially due to OOM in thread or improper execution.")

        return outputs_1, outputs_2, labels, ids
        
    def _store_probabilities(self, outputs_1, outputs_2, labels, ids, epoch: int):
        """Store probabilities for auto-labeled data."""
        probs_1 = torch.softmax(outputs_1, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        probs_2 = torch.softmax(outputs_2, dim=-1).gather(1, labels.to(self.device_2).unsqueeze(1)).squeeze(1)

        probs_1 = probs_1.cpu().tolist()
        probs_2 = probs_2.cpu().tolist()

        #print(f"------------------ ids = {len(ids)} \n prob1 = {len(probs_1)} -------------\n prob2 = {len(probs_2)} \n")

        if torch.is_tensor(ids):
            ids = ids.cpu().tolist()

        for id, prob1, prob2 in zip(ids, probs_1, probs_2):
            row = self.id2index_auto_label[id]
            #print(f"id = {id} and row = {row}\n")
            self.all_epoch_probabilities_1[row, epoch] = prob1
            self.all_epoch_probabilities_2[row, epoch] = prob2
            
    def _validate(self) -> float:
        """Run validation and return F1 score."""
        self.model_1.eval()
        self.model_2.eval()
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in self.dataloaders['val_dataloader']:
                batch_1 = {k: v.to(self.device_1) for k, v in batch.items()}
                outputs_1 = self.model_1(input_ids=batch_1['input_ids'], attention_mask=batch_1['attention_mask'])
                val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1)
                
                batch_2 = {k: v.to(self.device_2) for k, v in batch.items()}
                outputs_2 = self.model_2(input_ids=batch_2['input_ids'], attention_mask=batch_2['attention_mask'])
                val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1)
                
                val_probs = val_probs_1.cpu() + val_probs_2.cpu()
                out_ensembled = torch.argmax(val_probs, dim=1).cpu().numpy()
                
                y_pred.extend(out_ensembled.tolist())
                y_true.extend(batch_1['labels'].cpu().numpy().tolist())
                
        return f1_score(y_true, y_pred, average='macro')
        
    def _calculate_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate final weights based on probabilities."""
        slice_1 = self.all_epoch_probabilities_1[:, :self.best_epoch + 1]
        slice_2 = self.all_epoch_probabilities_2[:, :self.best_epoch + 1]

        means_1 = np.mean(slice_1, axis=1)
        stds_1 = np.std(slice_1, axis=1)
        means_2 = np.mean(slice_2, axis=1)
        stds_2 = np.std(slice_2, axis=1)

        if self.metric_combination == 'cv':
            return means_1 + stds_1, means_2 - stds_2
        elif self.metric_combination == 'cc':
            return means_2, means_1
        else:
            raise ValueError(f"Unknown metric combination: {self.metric_combination}")
            
    def _prepare_final_dataframe(self, weights_1: np.ndarray, weights_2: np.ndarray) -> pd.DataFrame:
        """Prepare the final output dataframe."""
        df = self.auto_labeled_data.copy()
        
        # Reorder data to match probability storage order
        index2id = {v: k for k, v in self.id2index_auto_label.items()}
        ordered_ids = [index2id[i] for i in range(len(index2id))]
        df = df.set_index('id').loc[ordered_ids].reset_index()
        
        # Add all calculated fields
        scaler = MinMaxScaler()
        
        df['init_w1'] = weights_1
        df['init_w2'] = weights_2
        df['all_epoch_probabilities_1'] = [list(row) for row in self.all_epoch_probabilities_1]
        df['all_epoch_probabilities_2'] = [list(row) for row in self.all_epoch_probabilities_2]
        df['best_epoch_probabilities_1'] = self.all_epoch_probabilities_1[:, self.best_epoch]
        df['best_epoch_probabilities_2'] = self.all_epoch_probabilities_2[:, self.best_epoch]
        df['init_w1_normalized'] = scaler.fit_transform(df[['init_w1']])
        df['init_w2_normalized'] = scaler.fit_transform(df[['init_w2']])
        
        return df
        
    def generate_weights(self) -> pd.DataFrame:
        """Main method to generate weights through training."""
        st = time.time()
        
        for epoch in range(self.training_params['num_epochs']):
            self.model_1.train()
            self.model_2.train()
            
            # Training loop
            for i, (batch_1, batch_2) in enumerate(zip(
                self.dataloaders['train_dataloader_1'], 
                self.dataloaders['train_dataloader_2']
            )):
                self._train_models(batch_1, batch_2)
                self._update_models(i, len(self.dataloaders['train_dataloader_1']))
            
            # Auto-label data probabilities
            self.model_1.eval()
            self.model_2.eval()
            
            #print(f"dataloaders['auto_label_dataloader'] = {len(self.dataloaders['auto_label_dataloader'])}")
            with torch.no_grad():
                for batch in self.dataloaders['auto_label_dataloader']:
                    outputs_1, outputs_2, labels, ids = self._parallel_inference(batch)
                    self._store_probabilities(outputs_1, outputs_2, labels, ids, epoch)
            
            # Validation
            cur_f1 = self._validate()
            epoch_time = time.time() - st
            st = time.time()
            
            log_message(f'Time taken for Epoch {epoch + 1}:{epoch_time:.2f} - F1: {cur_f1:.4f}', self.args)
            
            # Log to wandb if available
            if hasattr(self.args, 'wandb_exp') and self.args.wandb_exp:
                self.args.wandb_exp.log({"weight_gen_epoch": epoch + 1, "weight_gen_f1": cur_f1, "weight_gen_time": epoch_time})

            if cur_f1 > self.best_f1:
                self.best_f1 = cur_f1
                self.best_epoch = epoch
        
        log_message(f'Best F1:{self.best_f1:.4f} - Best Epoch:{self.best_epoch + 1}', self.args)
        
        # Log to wandb if available
        if hasattr(self.args, 'wandb_exp') and self.args.wandb_exp:
            self.args.wandb_exp.log({"weight_gen_best_f1": self.best_f1, "weight_gen_best_epoch": self.best_epoch + 1})
        
        # Calculate and return final weights
        weights_1, weights_2 = self._calculate_weights()
        return self._prepare_final_dataframe(weights_1, weights_2)




class CoTrainer:
    def __init__(self, args, models: Dict, dataloaders: Dict, 
                 training_params: Dict, optimizer_params: Dict, hyper_params: Dict,
                 devices: List, init_df: pd.DataFrame):
        """
        Initialize the CoTrainer with asynchronous parallel training capability.
        """
        self.args = args
        self.saved_model_dir = args.saved_model_dir
        self.model_1 = models['model_1']
        self.model_2 = models['model_2']
        self.dataloaders = dataloaders
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.hyper_params = hyper_params
        self.device_1 = devices[0]
        self.device_2 = devices[1]
        self.init_df = init_df
        self.metric_combination = args.metric_combination
        self.saved_model_name_suffix = args.saved_model_name_suffix
        # self.logger = logger
        # self.no_co_training = no_co_training
        
        # Thread synchronization primitives
        self.lock = threading.Lock()
        self.epoch_complete = threading.Event()
        self.weights_updated = threading.Event()
        
        self._initialize_models()
        self._initialize_training_state()
        
    def _initialize_models(self):
        """Move models to their respective devices."""
        self.model_1.to(self.device_1)
        self.model_2.to(self.device_2)
        
    def _initialize_training_state(self):
        """Initialize all training state variables."""
        self.train_weights_1 = np.array(self.init_df['init_w1'].values)
        self.train_weights_2 = np.array(self.init_df['init_w2'].values)
        
        self.train_weights_1_raw = copy.deepcopy(self.train_weights_1)
        self.train_weights_2_raw = copy.deepcopy(self.train_weights_2)
        
        initial_probabilities_1 = self.init_df['best_epoch_probabilities_1'].values
        initial_probabilities_2 = self.init_df['best_epoch_probabilities_2'].values
        
        num_samples = len(initial_probabilities_1)
        num_epochs = self.training_params['num_epochs']
        
        self.train_probabilities_all_epochs_1 = np.zeros((num_samples, num_epochs + 1))
        self.train_probabilities_all_epochs_2 = np.zeros((num_samples, num_epochs + 1))
        self.train_probabilities_all_epochs_1[:, 0] = initial_probabilities_1
        self.train_probabilities_all_epochs_2[:, 0] = initial_probabilities_2
        
        self.id2index_init_df = self._create_id_to_index_mapping(self.init_df)
        
        # Training state
        self.best_f1 = -1
        self.best_epoch = 0
        self.not_improving_epochs = 0
        self.best_epoch_probabilities_1 = [0] * num_samples
        self.best_epoch_probabilities_2 = [0] * num_samples
        
        # Thread-safe storage for batch results
        self.batch_results_1 = []
        self.batch_results_2 = []
    
    @staticmethod
    def _create_id_to_index_mapping(df) -> Dict:
        """Create a mapping from ID to global index."""
        return {row['id']: idx for idx, row in df.iterrows()}
    
    @staticmethod
    def _max_min_normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array values between 0 and 1 using min-max scaling."""
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.ones_like(arr) * 0.5
        return (arr - min_val) / (max_val - min_val)
    
    def _train_model(self, model, device, optimizer, is_model_1: bool):
        """Training function for each model (runs in separate thread)."""
        model.train()
        criterion = self.optimizer_params['criterion']
        accumulation_steps = self.training_params['accumulation_steps']
        
        for i, batch in enumerate(self.dataloaders['init_df_dataloader']):
            # Prepare batch for current model
            current_batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get rows and weights for current batch
            ids = batch['id'].cpu().tolist()
            rows = [self.id2index_init_df[guid] for guid in ids]
            
            with self.lock:
                weights = torch.tensor(
                    self.train_weights_1[rows] if is_model_1 else self.train_weights_2[rows], 
                    dtype=torch.double
                ).to(device)
            
            # Forward pass
            outputs = self._forward_pass(model, current_batch)
            
            # Compute loss
            loss = self._compute_loss(criterion, outputs, current_batch['labels'], weights, accumulation_steps)
            
            # Backward pass and optimization
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.dataloaders['init_df_dataloader']):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_params.get('max_grad_norm', 1.0))
                optimizer.step()
                optimizer.zero_grad()
            
            # Store results for weight updates
            with self.lock:
                if is_model_1:
                    self.batch_results_1.append({
                        'rows': rows,
                        'outputs': outputs.detach(),
                        'labels': current_batch['labels'].detach()
                    })
                else:
                    self.batch_results_2.append({
                        'rows': rows,
                        'outputs': outputs.detach(),
                        'labels': current_batch['labels'].detach()
                    })
        
        # Signal completion
        self.epoch_complete.set()
    
    def _update_weights_based_on_results(self, epoch: int):
        """Update weights based on results from both models."""
        epoch_idx = epoch + 1
        
        # Process model 1 results
        for result in self.batch_results_1:
            rows = result['rows']
            outputs = result['outputs']
            labels = result['labels']
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            golds = labels.cpu().numpy()
            probs_gold = np.array([probabilities[i][golds[i]] for i in range(len(golds))])
            
            self.train_probabilities_all_epochs_1[rows, epoch_idx] = probs_gold
        
        # Process model 2 results
        for result in self.batch_results_2:
            rows = result['rows']
            outputs = result['outputs']
            labels = result['labels']
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            golds = labels.cpu().numpy()
            probs_gold = np.array([probabilities[i][golds[i]] for i in range(len(golds))])
            
            self.train_probabilities_all_epochs_2[rows, epoch_idx] = probs_gold
        
        # Update weights for all samples
        for rows in [result['rows'] for result in self.batch_results_1]:
            probs_1_so_far = self.train_probabilities_all_epochs_1[rows, :epoch_idx + 1]
            probs_2_so_far = self.train_probabilities_all_epochs_2[rows, :epoch_idx + 1]
            
            mean_1 = np.mean(probs_1_so_far, axis=1)
            std_1 = np.std(probs_1_so_far, axis=1)
            mean_2 = np.mean(probs_2_so_far, axis=1)
            std_2 = np.std(probs_2_so_far, axis=1)

            if self.metric_combination == 'cv':
                if self.args.no_co_training:
                    self.train_weights_1[rows] = mean_1 + std_1
                    self.train_weights_2[rows] = mean_2 - std_2
                else:
                    self.train_weights_1[rows] = mean_2 - std_2
                    self.train_weights_2[rows] = mean_1 + std_1
            elif self.metric_combination == 'cc':
                self.train_weights_1[rows] = mean_2
                self.train_weights_2[rows] = mean_1
        
        # Clear results for next epoch
        self.batch_results_1.clear()
        self.batch_results_2.clear()
        
        # Normalize weights
        # self.train_weights_1 = self._max_min_normalize(self.train_weights_1)
        # self.train_weights_2 = self._max_min_normalize(self.train_weights_2)
        
        # Signal that weights are updated
        self.weights_updated.set()
        
    
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch using asynchronous parallel training."""
        # Reset synchronization flags
        self.epoch_complete.clear()
        self.weights_updated.clear()
        
        # Clear previous results
        self.batch_results_1.clear()
        self.batch_results_2.clear()
        
        # Create and start training threads
        thread1 = threading.Thread(
            target=self._train_model,
            args=(self.model_1, self.device_1, self.optimizer_params['optimizer_1'], True)
        )
        
        thread2 = threading.Thread(
            target=self._train_model,
            args=(self.model_2, self.device_2, self.optimizer_params['optimizer_2'], False)
        )
        
        thread1.start()
        thread2.start()
        
        # Wait for both models to complete their epoch
        self.epoch_complete.wait()
        
        # Update weights based on both models' results
        self._update_weights_based_on_results(epoch)
        
        # Wait for threads to finish
        thread1.join()
        thread2.join()
        
    def _forward_pass(self, model, batch):
        """Perform forward pass and get logits."""
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    
    def _compute_loss(self, criterion, outputs, labels, weights, accumulation_steps):
        """Compute weighted loss."""
        loss = criterion(outputs, labels) * weights
        loss = torch.mean(loss) / accumulation_steps
        return loss
    
    def _validate(self) -> float:
        """Perform validation and return F1 score."""
        self.model_1.eval()
        self.model_2.eval()
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in self.dataloaders['val_dataloader']:
                # Get predictions from both models
                outputs_1 = self._forward_pass(self.model_1, {k: v.to(self.device_1) for k, v in batch.items()})
                outputs_2 = self._forward_pass(self.model_2, {k: v.to(self.device_2) for k, v in batch.items()})
                
                # Ensemble predictions
                val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1).cpu()
                val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1).cpu()
                val_probs = val_probs_1 + val_probs_2
                
                # Get predictions and true labels
                out_ensembled = torch.argmax(val_probs, dim=1).numpy()
                y_pred.extend(out_ensembled.tolist())
                y_true.extend(batch['labels'].cpu().numpy().tolist())
        
        return f1_score(y_true, y_pred, average='macro')
    
    def _save_models(self):
        """Save the current best models."""
        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir)
            
        torch.save(self.model_1.state_dict(), 
                  os.path.join(self.saved_model_dir, f'co_trained_model_1{self.saved_model_name_suffix}.pt'))
        torch.save(self.model_2.state_dict(), 
                  os.path.join(self.saved_model_dir, f'co_trained_model_2{self.saved_model_name_suffix}.pt'))
    
    def train(self) -> pd.DataFrame:
        """Run the co-training process with asynchronous parallel training."""
        num_epochs = self.training_params['num_epochs']
        epoch_patience = self.hyper_params['EPOCH_PATIENCE']
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            if self.not_improving_epochs == epoch_patience:
                msg = f'Performance not improving for {epoch_patience} consecutive epochs.'
                print(msg)
                log_message(message=msg, args=self.args)
                break
            
            # Train for one epoch (asynchronously)
            self._train_epoch(epoch)
            
            # Validate
            current_f1 = self._validate()
            epoch_time = time.time() - start_time
            start_time = time.time()
            
            # Log progress
            msg = f'Time taken for Epoch {epoch + 1}: {epoch_time:.2f}s - F1: {current_f1:.8f}'
            log_message(message=msg, args=self.args)
            
            # Log to wandb if available
            if hasattr(self.args, 'wandb_exp') and self.args.wandb_exp:
                self.args.wandb_exp.log({"co_train_epoch": epoch + 1, "co_train_f1": current_f1, "co_train_time": epoch_time})
            
            # Check for improvement
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_epoch = epoch
                self.not_improving_epochs = 0
                
                self._save_models()
                self.best_epoch_probabilities_1[:] = self.train_probabilities_all_epochs_1[:, epoch + 1]
                self.best_epoch_probabilities_2[:] = self.train_probabilities_all_epochs_2[:, epoch + 1]
            else:
                self.not_improving_epochs += 1
            
            # Normalize weights
            self.train_weights_1 = self._max_min_normalize(self.train_weights_1)
            self.train_weights_2 = self._max_min_normalize(self.train_weights_2)
        
        df = copy.deepcopy(self.init_df)
        df['train_weights_1'] = self.train_weights_1
        df['train_weights_2'] = self.train_weights_2
        df['best_epoch_probabilities_1'] = self.best_epoch_probabilities_1
        df['best_epoch_probabilities_2'] = self.best_epoch_probabilities_2
        return df
    




class DualModelTrainer:
    def __init__(self, args, dataloaders: Dict, training_params: Dict,
                 optimizer_params: Dict, hyper_params: Dict, devices: Tuple[str, str],
                 models: Tuple[torch.nn.Module, torch.nn.Module]):
        """
        Initialize the ModelTrainer with all training components.
        """
        self.args = args
        self.saved_model_dir = args.saved_model_dir
        self.dataloaders = dataloaders
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.hyper_params = hyper_params
        self.device_1, self.device_2 = devices
        self.model_1, self.model_2 = models
        self.saved_model_name_suffix = args.saved_model_name_suffix
        
        
        self._initialize_models()
        self._initialize_training_state()


    def _initialize_models(self):
        """Move models to their respective devices and initialize training state."""
        self.model_1.to(self.device_1)
        self.model_2.to(self.device_2)
        
        self.optimizer_1 = self.optimizer_params['optimizer_1']
        self.optimizer_2 = self.optimizer_params['optimizer_2']
        self.lr_scheduler_1 = self.optimizer_params['lr_scheduler_1']
        self.lr_scheduler_2 = self.optimizer_params['lr_scheduler_2']
        self.criterion = self.optimizer_params['criterion']

    def _initialize_training_state(self):
        """Initialize training state variables."""
        self.best_f1 = -1
        self.best_epoch = 0
        self.not_improving_epochs = 0
        self.epoch_patience = self.hyper_params['EPOCH_PATIENCE']

    def _forward_pass(self, model, batch, device: str):
        """Perform forward pass and get logits."""
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        return outputs.logits if hasattr(outputs, 'logits') else outputs, batch['labels']

    def _compute_loss(self, outputs, labels, accumulation_steps: int):
        """Compute and scale loss."""
        return torch.mean(self.criterion(outputs, labels)) / accumulation_steps

    def _update_model(self, model, optimizer, lr_scheduler):
        """Update model parameters with gradient clipping."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_params.get('max_grad_norm', 1.0))
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    def _validate(self) -> float:
        """Run validation and return F1 score."""
        self.model_1.eval()
        self.model_2.eval()
        
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in self.dataloaders['val_dataloader']:
                # Get predictions from both models
                outputs_1, _ = self._forward_pass(self.model_1, batch, self.device_1)
                val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1)
                
                outputs_2, _ = self._forward_pass(self.model_2, batch, self.device_2)
                val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1)
                
                # Ensemble predictions
                val_probs = val_probs_1.cpu() + val_probs_2.cpu()
                predictions = torch.argmax(val_probs, dim=1).cpu().numpy()
                
                y_pred.extend(predictions.tolist())
                y_true.extend(batch['labels'].cpu().numpy().tolist())
        
        return f1_score(y_true, y_pred, average='macro')

    def _save_models(self):
        """Save the current best models."""
        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir)
            
        torch.save(self.model_1.state_dict(), 
                 os.path.join(self.saved_model_dir, f'final_model_1{self.saved_model_name_suffix}.pt'))
        torch.save(self.model_2.state_dict(), 
                 os.path.join(self.saved_model_dir, f'final_model_2{self.saved_model_name_suffix}.pt'))


    def _train_epoch(self, epoch: int):
        """Train for one epoch with batch processing kept internal."""
        self.model_1.train()
        self.model_2.train()
        
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        
        train_dl_1 = self.dataloaders['train_dataloader_1']
        train_dl_2 = self.dataloaders['train_dataloader_2']
        accumulation_steps = self.training_params['accumulation_steps']
        
        for batch_idx, (batch_1, batch_2) in enumerate(zip(train_dl_1, train_dl_2)):
            # Forward pass and loss computation for both models
            outputs_1, labels_1 = self._forward_pass(self.model_1, batch_1, self.device_1)
            outputs_2, labels_2 = self._forward_pass(self.model_2, batch_2, self.device_2)
            
            loss_1 = self._compute_loss(outputs_1, labels_1, accumulation_steps)
            loss_2 = self._compute_loss(outputs_2, labels_2, accumulation_steps)
            
            loss_1.backward()
            loss_2.backward()

            # Update models if accumulation steps reached
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dl_1):
                self._update_model(self.model_1, self.optimizer_1, self.lr_scheduler_1)
                self._update_model(self.model_2, self.optimizer_2, self.lr_scheduler_2)

    def train(self):
        """Main training loop."""
        start_time = time.time()
        
        for epoch in range(self.training_params['num_epochs']):
            if self.not_improving_epochs == self.epoch_patience:
                msg = f'Performance not improving for {self.epoch_patience} consecutive epochs.'
                log_message(msg, self.args)
                break
            
            self._train_epoch(epoch)
            current_f1 = self._validate()
            epoch_time = time.time() - start_time
            start_time = time.time()
            
            msg = f'Time taken for Epoch {epoch + 1}:{epoch_time:.2f} - F1: {current_f1:.4f}'
            log_message(msg, self.args)
            
            # Log to wandb if available
            if hasattr(self.args, 'wandb_exp') and self.args.wandb_exp:
                self.args.wandb_exp.log({"fine_tune_epoch": epoch + 1, "fine_tune_f1": current_f1, "fine_tune_time": epoch_time})
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_epoch = epoch
                self.not_improving_epochs = 0
                self._save_models()
            else:
                self.not_improving_epochs += 1
        
        msg = f'Best F1:{self.best_f1:.4f} - Best Epoch:{self.best_epoch}'
        log_message(msg, self.args)
        
        # Log to wandb if available
        if hasattr(self.args, 'wandb_exp') and self.args.wandb_exp:
            self.args.wandb_exp.log({"fine_tune_best_f1": self.best_f1, "fine_tune_best_epoch": self.best_epoch})


