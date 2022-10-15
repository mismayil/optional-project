import torch
from torch.utils.data import Dataset

class GlucoseDataset(Dataset):
    def __init__(self, data, tokenizer, source_col="input", target_col="target", max_source_len=512, max_target_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_col = source_col
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.tokenizer(
            [self.data.loc[index, self.source_col]],
            padding="max_length",
            max_length=self.max_source_len,
            return_tensors="pt",
            truncation=True,
        )
        target = self.tokenizer(
            [self.data.loc[index, self.target_col]],
            padding="max_length",
            max_length=self.max_target_len,
            return_tensors="pt",
            truncation=True,
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "labels": target_ids.to(dtype=torch.long),
        }