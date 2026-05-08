import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer, max_seq_len=128, labeled=True):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labeled = labeled
        self.idxes = list(range(len(text_list)))
        self.weights = [1] * len(self.text_list)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=self.max_seq_len, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        if self.labeled == True:
            item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['idx'] = self.idxes[idx]
        item['weights'] = self.weights[idx]
        return item

    def __len__(self):
        return len(self.text_list)


    def get_subset_dataset(self, idxs):
        text_lists = [self.text_list[i] for i in idxs]
        if self.labeled:
            label_lists = [self.labels[i] for i in idxs]
        else:
            label_lists = None
        return CustomDataset(text_lists, label_lists, self.tokenizer, self.max_seq_len, self.labeled)


class CustomDataset_tracked(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, idxes, tokenizer, max_seq_len=128, labeled=True):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labeled = labeled
        self.idxes = idxes
        self.weights = [1] * len(self.text_list)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=self.max_seq_len, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        if self.labeled == True:
            item['lbl'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['idx'] = self.idxes[idx]
        item['weights'] = self.weights[idx]
        return item

    def __len__(self):
        return len(self.text_list)


    def get_subset_dataset(self, idxs): 
        text_lists = [self.text_list[i] for i in idxs]
        label_lists = [self.labels[i] for i in idxs]
        id_lists = [self.idxes[i] for i in idxs]

        return CustomDataset_tracked(text_lists, label_lists, id_lists, self.tokenizer, self.max_seq_len, self.labeled)