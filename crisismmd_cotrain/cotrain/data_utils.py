import os
import pandas as pd
import numpy as np

def get_exponential_decay_ratio(num_classes, imbalance_ratio=10):
    cls_indices = np.arange(num_classes)
    ratios = imbalance_ratio ** (-cls_indices / (num_classes - 1))
    return ratios / ratios.sum()

def load_imb_dataset_helper(dataset, N, pseudo_label_shot, processed_dir, data_dir, use_correct_labels_only=None, mnli_split=None):
    """Helper function to load datasets with long-tailed label distribution."""

    def json2pd(filepath):
        return pd.read_json(filepath, orient='index')

    # Load test and validation sets
    testingSet = json2pd(os.path.join(data_dir, dataset, 'test.json'))
    validationSet = json2pd(os.path.join(data_dir, dataset, 'dev.json'))

    # Load full LLM-labeled training set
    llm_labeled_traininSet = json2pd(os.path.join(processed_dir, f'llm_labeled_trainingSet_{pseudo_label_shot}_shot.json'))
    llm_labeled_traininSet = llm_labeled_traininSet.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Get number of classes from y_true
    all_labels = llm_labeled_traininSet['ori_label'].unique()
    num_classes = len(all_labels)

    # Create long-tailed label distribution
    long_tail_ratio = get_exponential_decay_ratio(num_classes, imbalance_ratio=10)
    print(f"Long tail ratio: {long_tail_ratio}")
    
    samples_per_class = (N * 2 * long_tail_ratio).astype(int)
    samples_per_class = np.maximum(samples_per_class, 1)

    # Sample trainingSet_1 and trainingSet_2 from each class
    training_1_list, training_2_list = [], []
    used_ids = set()

    for i, label in enumerate(all_labels):
        class_df = llm_labeled_traininSet[llm_labeled_traininSet['ori_label'] == label]
        n_samples = samples_per_class[i]
        selected = class_df.head(n_samples)
        split_point = n_samples // 2

        training_1_list.append(selected.iloc[:split_point])
        training_2_list.append(selected.iloc[split_point:])
        used_ids.update(selected.index.tolist())

    trainingSet_1 = pd.concat(training_1_list).copy()
    trainingSet_2 = pd.concat(training_2_list).copy()

    # Now subsample auto_labeled_data from remaining using same long_tail_ratio
    remaining = llm_labeled_traininSet[~llm_labeled_traininSet.index.isin(used_ids)].copy()
    remaining = remaining[remaining['gen_label'] >= 0]  # ensure valid gen labels

    auto_labeled_list = []
    for i, label in enumerate(all_labels):
        class_df = remaining[remaining['ori_label'] == label]
        class_n = int(len(remaining) * long_tail_ratio[i])
        selected = class_df.head(class_n)
        auto_labeled_list.append(selected)

    auto_labeled_data = pd.concat(auto_labeled_list).copy()
    auto_labeled_data['label'] = auto_labeled_data['gen_label']
    trainingSet_1['label'] = trainingSet_1['ori_label']
    trainingSet_2['label'] = trainingSet_2['ori_label']

    # Optionally filter to only correct auto-labels
    if use_correct_labels_only:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] == auto_labeled_data['ori_label']]

    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data

def get_humaid_label_map():
    return {
        'requests_or_urgent_needs': 0,
        'rescue_volunteering_or_donation_effort': 1,
        'infrastructure_and_utility_damage': 2,
        'missing_or_found_people': 3,
        'displaced_people_and_evacuations': 4,
        'sympathy_and_support': 5,
        'injured_or_dead_people': 6,
        'caution_and_advice': 7,
        'other_relevant_information': 8,
        'not_humanitarian': 9
    }

def load_humaid_dataset(
    data_dir,
    pseudo_label_dir,
    event,
    lbcl,
    set_num,
    use_correct_labels_only=False,
):
    """
    Loads HumAID data from TSV files.
    """
    import pandas as pd
    
    # Define label map
    label_map = get_humaid_label_map()
    
    # Helper to load and process TSV
    def load_tsv(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath, sep='\t')
        
        # TODO: Schema naming is confusing (class_label=GOLD, label=PRED). Fix in future cleanup.
        
        # Map labels
        # 'class_label' is GOLD string
        if 'class_label' in df.columns:
            df['ori_label'] = df['class_label'].map(label_map).fillna(-1).astype(int)
        else:
             df['ori_label'] = -1
             
        # 'label' is PRED string (pseudo-label)
        if 'label' in df.columns:
             # Check if 'label' is string or int. If string, map it.
             # The user said "label is the pred label (string)"
             # But let's safely check just in case
             if df['label'].dtype == object:
                df['gen_label'] = df['label'].map(label_map).fillna(-1).astype(int)
             else:
                df['gen_label'] = df['label']
        else:
            df['gen_label'] = -1

        # Rename for consistency
        if 'tweet_id' in df.columns:
            df['id'] = df['tweet_id']
            
        return df

    # Load Standard Dev/Test (Original JSONs if they exist, or maybe mapped from TSV?)
    # The user transferred 'text_only.json' to 'data/humaid/dev' in previous steps?
    # Wait, the PLAN said we transferred the 'humaid' directory from remote.
    # The user just extracted the 'all_events.tsv' from 'anh_4o'.
    # We should assume standard dev/test exist in JSON or we use parts of all_events?
    # The original structure expects dev.json/test.json. 
    # Let's fallback to the generic helper for dev/test if they are standard JSONs, 
    # BUT if we are using all_events.tsv for everything, we might need logic here.
    # For now, let's assume dev/test are standard keys derived in load_dataset_helper 
    # and this function is specifically for the pseudo-label part or we override everything.
    
    # Actually, let's keep it simple: generic helper calls this for HumAID?
    # No, generic helper logic is messy. Let's try to feed back into generic helper 
    # OR just return the tuple expected by main.
    
    # Let's assume loading validation/test from the standard locations:
    valid_path = os.path.join(data_dir, "humaid", "joined", "dev.tsv")
    test_path = os.path.join(data_dir, "humaid", "joined", "test.tsv")
    
    validationSet = pd.read_csv(valid_path, sep='\t') if os.path.exists(valid_path) else pd.DataFrame()
    testingSet = pd.read_csv(test_path, sep='\t') if os.path.exists(test_path) else pd.DataFrame()
    
    validationSet['label'] = validationSet['class_label'].map(label_map)
    testingSet['label'] = testingSet['class_label'].map(label_map)

    print("Validation Set Columns:", validationSet.columns)
    print(validationSet.head())
    print("Testing Set Columns:", testingSet.columns)
    print(testingSet.head())


    # Filter by event
    if event:
        if 'event' in validationSet.columns:
            validationSet = validationSet[validationSet['event'] == event]
        if 'event' in testingSet.columns:
             testingSet = testingSet[testingSet['event'] == event]
    
    # Fix labels for dev/test
    if 'label' in validationSet.columns and validationSet['label'].dtype == object:
         validationSet['label'] = validationSet['label'].map(label_map)
    if 'label' in testingSet.columns and testingSet['label'].dtype == object:
         testingSet['label'] = testingSet['label'].map(label_map)

    # Load Pseudo-labels from all_events.tsv (anh_4o)
    # The user said: "focus on using the labels from the anh_4o folder (where all data with pseudolabels are in all_events.tsv)"
    # This implies this ONE file might contain everything or just the training pool?
    # "raw has each event split, joined has each split containing all events"
    # We will use all_events.tsv as the source for "auto_labeled_data" and "llm_labeled_traininSet"
    
    pseudo_path = os.path.join(data_dir, "humaid", pseudo_label_dir, "sep", f"{lbcl}lb", set_num, "unlabeled.tsv") 
    gold_path = os.path.join(data_dir, "humaid", pseudo_label_dir, "sep", f"{lbcl}lb", set_num, "labeled.tsv")

    plabel_df = load_tsv(pseudo_path)
    gold_df = load_tsv(gold_path)
    
    # Process Auto-Labeled Data (Pool)
    auto_labeled_data = plabel_df.copy()
    auto_labeled_data['label'] = auto_labeled_data['ori_label'] # Due to unlabeled.tsv using "class_label"
    if 'event' in auto_labeled_data.columns:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['event'] == event]
    
    # Process Gold Data (Initial Training Sets)
    gold_copy = gold_df.copy()
    gold_copy['label'] = gold_copy['ori_label']
    if 'event' in gold_copy.columns:
        gold_copy = gold_copy[gold_copy['event'] == event]
    
    # Shuffle the gold data
    gold_copy = gold_copy.sample(frac=1).reset_index(drop=True)

    half_point = len(gold_copy) // 2
    trainingSet_1 = gold_copy.iloc[:half_point].copy()
    trainingSet_2 = gold_copy.iloc[half_point:].copy()
    
    # TODO: Add use_correct_labels_only logic

    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data


def load_dataset_helper(
    use_correct_labels_only=None, 
    shots=None, 
    task_name=None, 
    data_dir=None, 
    pseudo_label_dir=None,
    event=None,
    lbcl=None,
    set_num=None
):
    """Helper function to load datasets based on dataset type."""
    
    if data_dir is None:
        raise ValueError("data_dir must be provided")
    if pseudo_label_dir is None:
        # Fallback default if not provided, though it's better to provide it
        pseudo_label_dir = os.path.join(data_dir, "pseudo_labels")

    def json2pd(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_json(filepath, orient='index')
    
    label_map = None
    if task_name == 'informative':
        label_map = {
            'not_informative': 0, 
            'informative': 1
        }
    elif task_name == 'humanitarian':
        label_map = {
            "affected_individuals": 0, 
            "rescue_volunteering_or_donation_effort": 1,
            "infrastructure_and_utility_damage": 2,
            "other_relevant_information": 3,
            "not_humanitarian": 4,
        }
    elif task_name == 'humaid':
        # Delegate to specific HumAID loader which handles TSV
        return load_humaid_dataset(data_dir, pseudo_label_dir, event, lbcl, set_num, use_correct_labels_only)

    
    # Construct paths dynamically
    # Expect data_dir/{task_name}/dev/text_only.json
    valid_json = os.path.join(data_dir, task_name, "dev", "text_only.json")
    validationSet = json2pd(valid_json)

    if label_map:
        validationSet['label'] = validationSet['label'].map(label_map)

    # Expect data_dir/{task_name}/test/text_only.json
    test_json = os.path.join(data_dir, task_name, "test", "text_only.json")
    testingSet = json2pd(test_json)

    if label_map:
        testingSet['label'] = testingSet['label'].map(label_map)

    # Handle pseudo-labels path construction
    # Original code had very specific hardcoded paths for CrisisMMD pseudo-labels
    # We will try to generalize: pseudo_label_dir/{task_name}/{shots}_shot_train_set1.json
    
    # For now, we will assume a simpler structure for the new setup, 
    # but try to support the legacy pathing if needed via the directory arguments.
    
    # NOTE: The original code used specific Llama-3.2 folders. 
    # We will assume pseudo_label_dir points to the PARENT of the specific experiment folder, 
    # OR we simplify and just look for the files directly in pseudo_label_dir for the given task.
    
    # Let's assume the files are directly in os.path.join(pseudo_label_dir, task_name)
    train_dir = os.path.join(pseudo_label_dir, task_name)
    
    # Determine filename pattern
    # The original was: info_{shots}-shot_train_set1.json or similar.
    # We'll use a standard f-string that can be adapted.
    
    # For HumAID, we might just use 'train_set1.json' etc if we create them manually.
    
    # Try to find the files
    try:
        if task_name == 'informative':
            prefix = 'info'
        elif task_name == 'humanitarian':
            prefix = 'huma' # Guessing prefix logic or just use task_name
        else:
            prefix = task_name

        # Construct filenames (trying to match roughly what was there but cleaned up)
        # Using specific logic for legacy compatibility if task is info/humanitarian
        if task_name in ['informative', 'humanitarian']:
             # Legacy path simulation if needed, or we just rely on the user moving files to match this structure
             # For this refactor, I will standardize:
             # {prefix}_{shots}-shot_train_set1.json
             train1_path = os.path.join(train_dir, f"{prefix}_{shots}-shot_train_set1.json")
             train2_path = os.path.join(train_dir, f"{prefix}_{shots}-shot_train_set2.json")
             unlabeled_path = os.path.join(train_dir, f"unlabeled_train_{shots}.json")
        else:
            # Standard path for new datasets (HumAID)
            train1_path = os.path.join(train_dir, f"train_set1_{shots}shot.json")
            train2_path = os.path.join(train_dir, f"train_set2_{shots}shot.json")
            unlabeled_path = os.path.join(train_dir, f"unlabeled_train_{shots}shot.json")
            
        trainingSet_1 = json2pd(train1_path)
        trainingSet_2 = json2pd(train2_path)
        auto_labeled_data = json2pd(unlabeled_path)
        
    except FileNotFoundError:
        # Fallback for "No Co-training" or "Dry Run" where these might not exist?
        # The main code crashes if they don't exist.
        print(f"WARNING: Could not find pseudo-label files in {train_dir}.")
        raise

    
    # Set labels appropriately
    trainingSet_1['label'] = trainingSet_1['ori_label']
    trainingSet_2['label'] = trainingSet_2['ori_label']
    auto_labeled_data['label'] = auto_labeled_data['gen_label']
    
    if use_correct_labels_only:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['gen_label'] == auto_labeled_data['ori_label']]
        
    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data
