from typing import List
import spacy
import torch
from torch.utils.data import Dataset
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")

doc_cache = {}
sent_cache = {}

def get_doc(text):
    if text not in doc_cache:
        doc = nlp(text)
        doc_cache[text] = doc

    return doc_cache[text]

def get_sents(text):
    if text not in sent_cache:
        doc = nlp(text)
        sent_cache[text] = [str(sent) for sent in doc.sents]

    return sent_cache[text]

with open("outputs/placeholders.txt") as f:
    placeholders = f.read().splitlines()

def get_words(text: str, ignore_stopwords: bool = True, ignore_entities: bool = True):
    doc = get_doc(text)
    tokens = [token for token in doc if token.text not in placeholders and not token.is_punct and len(token.text.strip()) > 0]
    
    if ignore_stopwords:
        tokens = [token for token in tokens if not token.is_stop]

    if ignore_entities:
        ents = [ent.text for ent in doc.ents]
        tokens = [token for token in tokens if token.text not in ents]

    return set([token.lemma_.lower() for token in tokens])

def has_rule_overlap(specific_text: str, general_text: str, threshold: float) -> bool:
    sp_words = get_words(specific_text)
    gl_words = get_words(general_text)

    if len(gl_words) > 0:
        return (len(sp_words.intersection(gl_words)) / len(gl_words)) > threshold
    
    return False

def has_context_overlap(words: List[str], context: str, threshold: float, ignore_stopwords: bool = True, ignore_entities: bool = True):
    context_words = get_words(context, ignore_stopwords=ignore_stopwords, ignore_entities=ignore_entities)

    if len(words) > 0 and (len(words.intersection(context_words)) / len(words)) > threshold:
        return True
    
    return False

def has_story_overlap(text: str, story: str, threshold: float, dim_based_overlap: bool = False, selected_context: str = None, dim: int = 1) -> bool:
    text_words = get_words(text)

    if dim_based_overlap and dim in [1, 6]:
        story_parts = re.split(re.escape(selected_context), story)
        past_context = story_parts[0]
        future_context = story_parts[1]
        context = None

        if dim == 1 and past_context:
            past_sents = get_sents(past_context)
            context = past_sents[-1]
        elif dim == 6 and future_context:
            future_sents = get_sents(future_context)
            context = future_sents[0]

        if context is not None:
            context_overlap = has_context_overlap(text_words, context, threshold=threshold)
            return context_overlap
    else:
        contexts = get_sents(story)
        for context in contexts:
            context_overlap = has_context_overlap(text_words, context, threshold=threshold)
            if context_overlap:
                return True

    return False

def has_story_copy(head: str, tail: str, story: str, threshold: float) -> bool:
    head_words = get_words(head, ignore_stopwords=False, ignore_entities=False)
    tail_words = get_words(tail, ignore_stopwords=False, ignore_entities=False)
    sentences = get_sents(story)

    for i, sentence in enumerate(sentences):
        head_overlap = has_context_overlap(head_words, sentence, threshold, ignore_stopwords=False, ignore_entities=False)
        if head_overlap:
            if i > 0:
                tail_overlap = has_context_overlap(tail_words, sentences[i-1], threshold, ignore_stopwords=False, ignore_entities=False)
                if tail_overlap:
                    return True, i, "backward"
            if i < len(sentences)-1:
                tail_overlap = has_context_overlap(tail_words, sentences[i+1], threshold, ignore_stopwords=False, ignore_entities=False)
                if tail_overlap:
                    return True, i, "forward"
    
    return False, 0, "none"


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