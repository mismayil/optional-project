from typing import List
import spacy
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import itertools
import json

nlp = spacy.load("en_core_web_sm")

doc_cache = {}
sent_cache = {}

def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def write_json(obj, filepath, indent=2):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=indent)

def get_doc(text):
    if text not in doc_cache:
        doc = nlp(text)
        doc_cache[text] = doc

    return doc_cache[text]

def get_sents(text, by_sentence=None, use_cache=True):
    if not use_cache or text not in sent_cache:
        if by_sentence:
            text_parts = re.split(re.escape(by_sentence), text)
            past_sents = get_sents(text_parts[0])
            future_sents = get_sents(text_parts[1])
            sent_cache[text] = list(itertools.chain(past_sents, [by_sentence], future_sents))
        else:
            doc = nlp(text.strip())
            sent_cache[text] = [sent.text for sent in doc.sents if len(sent.text.strip()) > 0]

    return sent_cache[text]

with open("outputs/placeholders.txt") as f:
    placeholders = f.read().splitlines()

def get_words(text: str, ignore_stopwords: bool = False, ignore_entities: bool = False):
    doc = get_doc(text.strip())
    tokens = [token for token in doc if token.text not in placeholders and not token.is_punct and len(token.text.strip()) > 0]
    
    if ignore_stopwords:
        tokens = [token for token in tokens if not token.is_stop]

    if ignore_entities:
        ents = [ent.text for ent in doc.ents]
        tokens = [token for token in tokens if token.text not in ents]

    return set([token.lemma_.lower() for token in tokens])

def has_overlap(source: str, target: str, threshold: float, ignore_stopwords: bool = False, ignore_entities: bool = False):
    source_words = get_words(source, ignore_stopwords=ignore_stopwords, ignore_entities=ignore_entities)
    target_words = get_words(target, ignore_stopwords=ignore_stopwords, ignore_entities=ignore_entities)

    if len(target_words) > 0:
        return (len(target_words.intersection(source_words)) / len(target_words)) > threshold
    
    return False

def has_story_overlap(head: str, tail: str, story: str, threshold: float, selected_context: str = None, dim: int = 1,
                      ignore_stopwords: bool = False, ignore_entities: bool = False) -> bool:
    story_parts = re.split(re.escape(selected_context), story)
    past_context = story_parts[0]
    future_context = story_parts[1]
    context = None
    x = None

    if dim <= 5 and past_context:
        past_sents = get_sents(past_context)
        context = past_sents[-1]
        x = head
    elif dim >= 6 and future_context:
        future_sents = get_sents(future_context)
        context = future_sents[0]
        x = tail

    if context is not None:
        context_overlap = has_overlap(context, x, threshold=threshold, ignore_stopwords=ignore_stopwords, ignore_entities=ignore_entities)
        return context_overlap

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