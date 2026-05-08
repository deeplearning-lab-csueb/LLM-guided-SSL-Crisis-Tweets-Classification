import torch
from torch.utils.data import Dataset
import re
import data_utils


class BaseDatasetProcessor:
    def process_dataframe(self, dataframe):
        raise NotImplementedError("Subclasses should implement this method")

    def extract_int_from_string(self, s):
        if isinstance(s, int):
            return s
        if isinstance(s, float):
            # You can choose how to handle floats: either convert if whole number, or ignore
            return int(s)
        if isinstance(s, str):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else None
        return None


class GenericLabelProcessor:
    def get_numeric_label(self, label, label_map):
        if isinstance(label, int) and label in label_map.values():
            return label
        elif isinstance(label, str) and label.isdigit():
            return int(label)
        return label_map.get(label, -1)

    def get_textual_label(self, label, idx_to_label):
        return idx_to_label.get(label, -1)


class TextOnlyProcessor(BaseDatasetProcessor, GenericLabelProcessor):
    def __init__(self, label_map=None):
        self.label_map = label_map if label_map is not None else {
            "affected_individuals": 0, 
            "rescue_volunteering_or_donation_effort": 1,
            "infrastructure_and_utility_damage": 2,
            "other_relevant_information": 3,
            "not_humanitarian": 4,
        }

    def process_dataframe(self, dataframe):
        
        if 'ori' in dataframe.columns:
            dataframe.rename(columns={'ori': 'sentence'}, inplace=True)
        
        if 'idx' in dataframe.columns:
            dataframe.rename(columns={'idx': 'id'}, inplace=True)
        
        dataframe['id'] = dataframe.get('tweet_id') if 'id' in dataframe.columns else dataframe.index
        dataframe['id'] = dataframe['tweet_id'].apply(self.extract_int_from_string)
        dataframe['tweet_text'] = dataframe['tweet_text']
        dataframe['label'] = dataframe['label']

        # print(f"Length of Dataframe ------- {len(dataframe)}")
        
        return_keys = ['id', 'tweet_text', 'label']

        # if 'ori_label' in dataframe.columns:
        #     dataframe['label'] = dataframe['ori_label'] #.apply(lambda l: self.get_numeric_label(l, self.label_map))
        #     return_keys.append('label')
            
        # if 'label' in dataframe.columns:
        #     dataframe['label'] = dataframe['label'].apply(lambda l: self.get_numeric_label(l, self.label_map))
        #     return_keys.append('label')

        #print(f"Return keys: {dataframe[return_keys]}")
        return dataframe[return_keys]

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, dataset='sci_nli', include_augmented=False):
        self.dataset = dataset
        self.encoder = TextEncoder(tokenizer, max_len)
        self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)
        self.include_augmented = include_augmented
        # print(f'df columns: {self.dataframe.columns}')
        if self.include_augmented:
            if self.dataset not in ['informative', 'humanitarian']:
                raise ValueError(f"Augmented data is only available for ag_news, yahoo_answers, amazon_review, yelp_review, and aclImdb datasets")
            if 'aug_1' not in self.dataframe.columns:
                raise ValueError(f"Augmented data requested but 'aug_1' column not found in {dataset} dataframe")
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        #print(f"-------------------\n row = {row}")
        #print(f"-------------------\n {row.keys()}")
        item = {
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'id': row['id']
        }
        if 'ori_label' in row:
            item['ori_labels'] = torch.tensor(row['label'], dtype=torch.long)

        input_ids, attention_mask = self.encoder.encode_sentence(
            str(row['tweet_text'])
        )
        
        
        if self.include_augmented:
            # Weak augmentation (aug_0)
            aug0_input_ids, aug0_token_type_ids, aug0_attention_mask = self.encoder.encode_sentence(
                str(row['aug_0'])
            )
            # Strong augmentation (aug_1)
            aug1_input_ids, aug1_token_type_ids, aug1_attention_mask = self.encoder.encode_sentence(
                str(row['aug_1'])
            )
            item.update({
                'aug0_input_ids': aug0_input_ids,
                'aug0_token_type_ids': aug0_token_type_ids,
                'aug0_attention_mask': aug0_attention_mask,
                'aug1_input_ids': aug1_input_ids,
                'aug1_token_type_ids': aug1_token_type_ids,
                'aug1_attention_mask': aug1_attention_mask,
            })

        # Add the common encoding fields
        item.update({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        })

        return item
    
    def get_dataset_processor(self, dataset):
        label_maps = {
            'informative': {'informative': 1, 'not_informative': 0},
            'humanitarian': {
                "affected_individuals": 0, 
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            },
        }
        processors = {
            'informative': TextOnlyProcessor(label_maps['informative']),
            'humanitarian': TextOnlyProcessor(label_maps['humanitarian']),
            'humaid': TextOnlyProcessor(data_utils.get_humaid_label_map()),
        }
        if dataset not in processors:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return processors[dataset]  


class TextEncoder:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_sentence(self, sentence):
        """Encode a single sentence and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_pair_inputs(self, sentence1, sentence2):
        """Encode a pair of sentences and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_mc_inputs(self, context, start_ending, endings):
        """Encode multiple choice inputs with context and multiple endings."""
#        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        all_input_ids, all_attention_masks = [], [], []
        
        for ending in endings:
            full_ending = f"{start_ending} {ending}" if start_ending else ending
            inputs = self.tokenizer.encode_plus(
                context,
                full_ending,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt'  # Return PyTorch tensors directly
            )
            
            all_input_ids.append(inputs['input_ids'].squeeze(0))
            #all_token_type_ids.append(inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0))
            all_attention_masks.append(inputs['attention_mask'].squeeze(0))
#        return torch.stack(all_input_ids), torch.stack(all_token_type_ids), torch.stack(all_attention_masks)            
        return torch.stack(all_input_ids), torch.stack(all_attention_masks)
    
class ImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, dataset='sci_nli', include_augmented=False):
        self.dataset = dataset
        self.encoder = TextEncoder(tokenizer, max_len)
        self.dataframe = self.get_dataset_processor(dataset).process_dataframe(dataframe)
        self.include_augmented = include_augmented
        # print(f'df columns: {self.dataframe.columns}')
        if self.include_augmented:
            if self.dataset not in ['ag_news', 'yahoo_answers', 'amazon_review', 'yelp_review', 'aclImdb', 'informative', 'humanitarian']:
                raise ValueError(f"Augmented data is only available for ag_news, yahoo_answers, amazon_review, yelp_review, and aclImdb datasets")
            if 'aug_1' not in self.dataframe.columns:
                raise ValueError(f"Augmented data requested but 'aug_1' column not found in {dataset} dataframe")
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        #print(f"-------------------\n row = {row}")
        #print(f"-------------------\n {row.keys()}")
        item = {
            'labels': torch.tensor(row['label'], dtype=torch.long),
            'id': row['id']
        }
        if 'ori_label' in row:
            item['ori_labels'] = torch.tensor(row['label'], dtype=torch.long)

        if self.dataset in ['informative', 'humanitarian']:
#            input_ids, token_type_ids, attention_mask = self.encoder.encode_sentence(
            input_ids, attention_mask = self.encoder.encode_sentence(
                str(row['tweet_text'])
            )
            
            
            if self.include_augmented:
                # Weak augmentation (aug_0)
                aug0_input_ids, aug0_token_type_ids, aug0_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_0'])
                )
                # Strong augmentation (aug_1)
                aug1_input_ids, aug1_token_type_ids, aug1_attention_mask = self.encoder.encode_sentence(
                    str(row['aug_1'])
                )
                item.update({
                    'aug0_input_ids': aug0_input_ids,
                    'aug0_token_type_ids': aug0_token_type_ids,
                    'aug0_attention_mask': aug0_attention_mask,
                    'aug1_input_ids': aug1_input_ids,
                    'aug1_token_type_ids': aug1_token_type_ids,
                    'aug1_attention_mask': aug1_attention_mask,
                })
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Add the common encoding fields
        item.update({
            'input_ids': input_ids,
            # 'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        })

        return item
    
    def get_dataset_processor(self, dataset):
        label_maps = {
            'informative': {'informative': 1, 'not_informative': 0},
            'humanitarian': {
                "affected_individuals": 0, 
                "rescue_volunteering_or_donation_effort": 1,
                "infrastructure_and_utility_damage": 2,
                "other_relevant_information": 3,
                "not_humanitarian": 4,
            },
            'humaid': data_utils.get_humaid_label_map(),
        }
        processors = {
            'aclImdb': TextOnlyProcessor(label_maps['aclImdb']),
            'informative': TextOnlyProcessor(label_maps['informative']),
            'humanitarian': TextOnlyProcessor(label_maps['humanitarian']),
            'humaid': TextOnlyProcessor(label_maps['humaid']),
        }
        if dataset not in processors:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return processors[dataset]  


class ImageEncoder:
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_sentence(self, sentence):
        """Encode a single sentence and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_pair_inputs(self, sentence1, sentence2):
        """Encode a pair of sentences and return tensors."""
        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'  # Return PyTorch tensors directly
        )
        return (
            inputs['input_ids'].squeeze(0),
            #inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0),
            inputs['attention_mask'].squeeze(0)
        )

    def encode_mc_inputs(self, context, start_ending, endings):
        """Encode multiple choice inputs with context and multiple endings."""
#        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        all_input_ids, all_attention_masks = [], [], []
        
        for ending in endings:
            full_ending = f"{start_ending} {ending}" if start_ending else ending
            inputs = self.tokenizer.encode_plus(
                context,
                full_ending,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt'  # Return PyTorch tensors directly
            )
            
            all_input_ids.append(inputs['input_ids'].squeeze(0))
            #all_token_type_ids.append(inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).squeeze(0))
            all_attention_masks.append(inputs['attention_mask'].squeeze(0))
#        return torch.stack(all_input_ids), torch.stack(all_token_type_ids), torch.stack(all_attention_masks)            
        return torch.stack(all_input_ids), torch.stack(all_attention_masks)
