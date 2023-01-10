import argparse
from tqdm import tqdm
import pathlib
from collections import defaultdict

from utils import (
    read_json, write_json
)

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--overlap-path", type=str, help="Path to overlap data")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")

    args = parser.parse_args()

    data = read_json(args.datapath)
    overlap_data = read_json(args.overlap_path)

    overlap_map = defaultdict(list)
    data_map = {}

    for pair in overlap_data:
        overlap_map[pair["id1"]].append(pair)
        overlap_map[pair["id2"]].append(pair)

    for sample in data:
        data_map[sample["id"]] = sample

    for sample in tqdm(data, total=len(data), desc="Extending"):
        overlap_pairs = overlap_map.get(sample["id"])
        exchanged_norms = 0
        exchanged_fake_norms = 0

        if overlap_pairs:
            for pair in overlap_pairs:
                peer_id = pair["id1"] if sample["id"] == pair["id2"] else pair["id2"]
                peer_sample = data_map[peer_id]

                if pair["norm_similarity"] > 0.5:
                    sample["other_norms"].append(peer_sample["norm"])
                    exchanged_norms += 1
                else:
                    sample["fake_norms"].append(peer_sample["norm"])
                    sample["fake_norm_sentiments"].append(peer_sample["norm_sentiment"])
                    sample["fake_norm_concepts"].append(peer_sample["norm_concepts"])
                    exchanged_fake_norms += 1
        
        sample["exchanged_norms"] = exchanged_norms
        sample["exchanged_fake_norms"] = exchanged_fake_norms

    datapath = pathlib.Path(args.datapath)
    write_json(data, f"{datapath.parent}/{datapath.stem}_augmented{args.suffix}.json")

if __name__ == "__main__":
    main()