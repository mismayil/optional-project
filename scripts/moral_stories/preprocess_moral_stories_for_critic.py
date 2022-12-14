import argparse
from tqdm import tqdm
import pathlib
import spacy
import random

from utils import read_jsonl, write_json, get_phrases, has_overlap

NORM_PREFIXES = [
    "It's good to",
    "It's bad to",
    "It's wrong to",
    "It's right to",
    "It's rude to",
    "You should",
    "You shouldn't"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--neg-datapath", type=str, help="Path to contrastive moral stories dataset")
    parser.add_argument("--concept-type", type=str, choices=["np", "vp"], default="vp", help="What type of concepts to extract from moral action and norms")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    neg_data = read_jsonl(args.neg_datapath)
    critic_data = []

    neg_data_cache = {}

    for neg_sample in neg_data:
        neg_data_cache[neg_sample["ID"]] = neg_sample

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        neg_sample = neg_data_cache.get(sample["ID"])
        situation = sample["situation"]
        intention = sample["intention"]
        moral_action = sample["moral_action"]
        immoral_action = sample["immoral_action"]
        norm = sample["norm"]
        anti_norm = neg_sample["norm"] if neg_sample else None
        context = f"{situation} {intention} {immoral_action}"
        context_phrases = get_phrases(context, "vp")
        moral_phrases = get_phrases(moral_action, args.concept_type)
        immoral_phrases = get_phrases(immoral_action, args.concept_type)
        norm_phrases = get_phrases(norm, args.concept_type)

        fake_norms = []
        discarded_norms = []

        for phrase in context_phrases:
            fake_norm = f"{random.choice(NORM_PREFIXES)} {phrase}"
            if has_overlap(phrase, norm, 0.5):
                discarded_norms.append(fake_norm)
            else:
                fake_norms.append(fake_norm)
        
        critic_data.append({
            "id": sample["ID"],
            "situation": situation,
            "intention": intention,
            "immoral_action": immoral_action,
            "norm": norm,
            "anti_norm": anti_norm,
            "fake_norms": fake_norms,
            "discarded_norms": discarded_norms,
            "moral_action": moral_action,
            "moral_action_concepts": list(moral_phrases),
            "immoral_action_concepts": list(immoral_phrases),
            "norm_concepts": list(norm_phrases)
        })

    datapath = pathlib.Path(args.datapath)
    dataset_type = "train"

    if "dev" in datapath.stem or "val" in datapath.stem:
        dataset_type = "dev"
    elif "test" in datapath.stem:
        dataset_type = "test"

    write_json(critic_data, f"{datapath.parent}/critic_{dataset_type}_prep_{args.concept_type}.json")

if __name__ == "__main__":
    main()