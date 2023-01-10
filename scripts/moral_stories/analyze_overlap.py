import argparse
from tqdm import tqdm
import pathlib

from utils import (
    read_json, write_json, get_words, get_overlap
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")

    args = parser.parse_args()

    data = read_json(args.datapath)

    words_map = []
    overlap_pairs = []

    for sample in tqdm(data, total=len(data), desc="Collecting words"):
        context = " ".join([sample["situation"], sample["intention"], sample["moral_action"], sample["immoral_action"]])
        words_map.append({
            "id": sample["id"],
            "context": context,
            "norm": sample["norm"], 
            "words": get_words(context, ignore_stopwords=True, ignore_entities=True)
        })

    for i, sample1 in tqdm(enumerate(words_map), total=len(words_map), desc="Analyzing"):
        for j, sample2 in enumerate(words_map[i+1:]):
            overlap = set(sample1["words"]).intersection(set(sample2["words"]))
            if len(overlap) / len(sample1["words"]) > 0.5 or len(overlap) / len(sample2["words"]) > 0.5:
                pair = {
                    "id1": sample1["id"], 
                    "id2": sample2["id"], 
                    "context1": sample1["context"], 
                    "context2": sample2["context"],
                    "norm1": sample1["norm"],
                    "norm2": sample2["norm"], 
                    "context_overlap": list(overlap)
                }
                norm_overlap, norm_overlap_level1, norm_overlap_level2 = get_overlap(sample1["norm"], sample2["norm"], ignore_entities=True, ignore_stopwords=True)
                pair["norm_similarity"] = max(norm_overlap_level1, norm_overlap_level2)
                pair["norm_overlap"] = list(norm_overlap)
                overlap_pairs.append(pair)

    datapath = pathlib.Path(args.datapath)
    write_json(overlap_pairs, f"outputs/{datapath.stem}_overlap{args.suffix}.json")

if __name__ == "__main__":
    main()