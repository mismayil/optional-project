import argparse
from tqdm import tqdm
import pathlib
from itertools import product
import spacy
import random

from utils import read_jsonl, write_json, get_verb_phrases, get_noun_phrases, has_overlap

NORM_PREFIXES = [
    "It's good",
    "It's bad",
    "It's wrong",
    "It's right"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--neg-datapath", type=str, help="Path to contrastive moral stories dataset")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    neg_data = read_jsonl(args.neg_datapath)
    critic_data = []

    nlp = spacy.load("en_core_web_sm")

    for sample, neg_sample in tqdm(zip(data, neg_data[::2]), total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        moral_action = sample["moral_action"]
        immoral_action = sample["immoral_action"]
        norm = sample["norm"]
        anti_norm = neg_sample["norm"]
        context = f"{situation} {intention} {immoral_action}"
        context_doc = nlp(context)
        moral_doc = nlp(moral_action)
        norm_doc = nlp(norm)

        moral_concepts = set()
        norm_concepts = set()

        context_phrases = []

        for token in context_doc:
            context_phrases.extend(get_verb_phrases(token))
        
        for token in moral_doc:
            moral_concepts.update(get_noun_phrases(token))

        for token in norm_doc:
            norm_concepts.update(get_noun_phrases(token))

        fake_norms = []
        discarded_norms = []

        for phrase in context_phrases:
            fake_norm = f"{random.choice(NORM_PREFIXES)} {phrase}"
            if has_overlap(phrase, norm, 0.5):
                discarded_norms.append(fake_norm)
            else:
                fake_norms.append(fake_norm)
        
        critic_data.append({
            "situation": situation,
            "intention": intention,
            "immoral_action": immoral_action,
            "norm": norm,
            "anti_norm": anti_norm,
            "fake_norms": fake_norms,
            "discarded_norms": discarded_norms,
            "moral_action": moral_action,
            "moral_concepts": list(moral_concepts),
            "norm_concepts": list(norm_concepts)
        })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/critic_with_neg_{datapath.stem}.json")

if __name__ == "__main__":
    main()