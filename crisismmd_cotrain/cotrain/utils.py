import pandas as pd
import os
import numpy as np
import argparse


def max_min_normalize_all_values(lst):
        min_val = min(lst)
        max_val = max(lst)
        return [(x - min_val) / (max_val - min_val) for x in lst]
    
def split_dataframe(train_df, n, random_seed):
    """
    Splits the given dataframe into two dataframes, each containing n samples from each class,
    and returns a third dataframe containing the rest of the samples.
    
    Parameters:
    - train_df (pd.DataFrame): The input dataframe containing the data with a 'label' column.
    - n (int): The number of samples per class for each resulting dataframe.
    - random_seed (int): The random seed for reproducibility.
    
    Returns:
    - df1 (pd.DataFrame): The first resulting dataframe with n samples from each class.
    - df2 (pd.DataFrame): The second resulting dataframe with n samples from each class.
    - df_rest (pd.DataFrame): The dataframe containing the rest of the samples not in df1 and df2.
    """
    
    # Shuffle the DataFrame with a fixed random seed
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Create two empty DataFrames
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    # Track the indices of selected samples
    selected_indices = []

    # Group by the class label
    grouped = train_df.groupby('label')

    # Sample 2n instances from each class and split into two dataframes with a fixed random seed
    for label, group in grouped:
        if len(group) >= 2 * n:
            sampled_group = group.sample(2 * n, random_state=random_seed)
            df1 = pd.concat([df1, sampled_group.iloc[:n]])
            df2 = pd.concat([df2, sampled_group.iloc[n:2*n]])
            selected_indices.extend(sampled_group.index[:2*n])
        else:
            print(f"Not enough samples for class {label}")

    # Reset index for the new dataframes
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Create a dataframe with the remaining samples
    df_rest = train_df.drop(index=selected_indices).reset_index(drop=True)

    return df1, df2, df_rest

# Example usage
# train_df = pd.read_csv('your_data.csv')
# df1, df2, df_rest = split_dataframe(train_df, n=10, random_seed=42)




def prepare_training_sets(data_dir, dataset, processed_dir, N, pseudo_label_shot):
    # Load datasets
    testingSet = pd.read_json(os.path.join(data_dir, dataset, 'test.json'), orient='index')
    validationSet = pd.read_json(os.path.join(data_dir, dataset, 'dev.json'), orient='index')
    
    train_1_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'train_1_ids.npy'))
    train_2_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'train_2_ids.npy'))
    auto_labeled_data_ids = np.load(os.path.join(data_dir, dataset, 'labeled_idx', f'N_{N}', 'auto_labeled_data_ids.npy'))
    
    if pseudo_label_shot == 0:
        llm_labeled_traininSet = pd.read_json(os.path.join(processed_dir, 'llm_labeled_trainingSet.json'), orient='index')
    else:
        llm_labeled_traininSet = pd.read_json(os.path.join(processed_dir, f'N_{N}', 'llm_labeled_trainingSet.json'), orient='index')
        
    # Prepare training sets
    trainingSet_1 = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(train_1_ids)].copy()
    trainingSet_2 = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(train_2_ids)].copy()
    auto_labeled_data = llm_labeled_traininSet[llm_labeled_traininSet['id'].isin(auto_labeled_data_ids)].copy()
    
    # Rename columns
    trainingSet_1['label'] = trainingSet_1['ori_label']
    trainingSet_2['label'] = trainingSet_2['ori_label']
    auto_labeled_data['label'] = auto_labeled_data['gen_label']
    
    trainingSet_1['sentence'] = trainingSet_1['ori']
    trainingSet_2['sentence'] = trainingSet_2['ori']
    auto_labeled_data['sentence'] = auto_labeled_data['ori']
    
    # Select relevant columns
    trainingSet_1 = trainingSet_1[['id', 'sentence', 'label']]
    trainingSet_2 = trainingSet_2[['id', 'sentence', 'label']]
    auto_labeled_data = auto_labeled_data[['id', 'sentence', 'label']]
    
    # Concatenate training sets
    all_trainingSet = pd.concat([trainingSet_1, trainingSet_2], ignore_index=True)
    
    return trainingSet_1, trainingSet_2, all_trainingSet, testingSet, validationSet, auto_labeled_data


def delete_saved_models(model_path):
    
    # Delete the files with error handling
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Successfully deleted {model_path}")
        else:
            print(f"File not found: {model_path}")
        
        

    except PermissionError:
        print("Permission denied: Unable to delete model files")
    except Exception as e:
        print(f"Error occurred while deleting model files: {str(e)}")
        
        
def log_message(message, args):
    """Log message to file and console."""
    print(message)
    if args.setup_local_logging:
        args.logger.info(message)
    if args.comet_ml:
        args.comet_exp.log_text(message)
        
def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
  
