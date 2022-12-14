import argparse
from tqdm import tqdm
import pathlib

from utils import read_json, write_json, get_phrases

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")

    args = parser.parse_args()

    data = read_json(args.datapath)

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        norm = sample["norm"]
        norm_concepts = sample["norm_concepts"]
        norm_noun_phrases = get_phrases(norm, "np")
        sample["norm_concepts"] = list(set(norm_concepts).union(set(norm_noun_phrases)))

    datapath = pathlib.Path(args.datapath)

    write_json(data, f"{datapath.parent}/{datapath.stem}_with_norm_np.json")

if __name__ == "__main__":
    main()