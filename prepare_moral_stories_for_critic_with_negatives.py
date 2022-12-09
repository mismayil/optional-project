import argparse
from tqdm import tqdm
import pathlib
from itertools import product
import spacy

from utils import read_jsonl, write_json, get_verb_phrases, has_overlap

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
        action = sample["immoral_action"]
        norm = sample["norm"]
        anti_norm = neg_sample["norm"]
        context = f"{situation} {intention} {action}"
        doc = nlp(context)

        context_phrases = []

        for token in doc:
            context_phrases.extend(get_verb_phrases(token))
        
        fake_norms = [f"{prefix} {phrase}" for prefix, phrase in product(NORM_PREFIXES, context_phrases) if not has_overlap(f"{prefix} {phrase}", norm, 0.5)]
        
        critic_data.append({
            "context": context,
            "norm": norm,
            "anti_norm": anti_norm,
            "fake_norms": fake_norms
        })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/critic_with_neg_{datapath.stem}.json")

if __name__ == "__main__":
    main()