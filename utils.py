import spacy
import torch
from torch.utils.data import Dataset
import pandas as pd

nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])

doc_cache = {}

def get_doc(text):
    if text in doc_cache:
        return doc_cache[text]
    
    doc = nlp(text)
    doc_cache[text] = doc

    return doc

with open("outputs/placeholders.txt") as f:
    placeholders = f.read().splitlines()

def has_overlap(specific_text: str, general_text: str, threshold: float) -> bool:
    sp_doc = get_doc(specific_text)
    gl_doc = get_doc(general_text)

    sp_words = set([token.lemma_.lower() for token in sp_doc])
    gl_words = set([token.lemma_.lower() for token in gl_doc if token.text not in placeholders])

    if len(gl_words) > 0:
        return (len(sp_words.intersection(gl_words)) / len(gl_words)) > threshold
    
    return False


def has_overlap_with_story(text: str, story: str, threshold: float) -> bool:
    contexts = story.split(".")
    doc = get_doc(text)
    words = set([token.lemma_.lower() for token in doc])

    for context in contexts:
        context_doc = get_doc(context)
        context_words = set([token.lemma_.lower() for token in context_doc])

        if len(words) > 0 and (len(words.intersection(context_words)) / len(words)) > threshold:
            return True

    return False

def get_glucose_general_data(glucose_data):
    glucose_gen_data = pd.DataFrame()
    glucose_gen_data["input"] = glucose_data.iloc[:, 0]
    glucose_gen_data["target"] = glucose_data.iloc[:, 1].apply(lambda x: x.split("**")[1].strip())
    return glucose_gen_data

def get_glucose_specific_data(glucose_data):
    glucose_spec_data = pd.DataFrame()
    glucose_spec_data["input"] = glucose_data.iloc[:, 0]
    glucose_spec_data["target"] = glucose_data.iloc[:, 1].apply(lambda x: x.split("**")[0].strip())
    return glucose_spec_data

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