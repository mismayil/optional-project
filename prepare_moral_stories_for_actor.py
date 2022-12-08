import argparse
from tqdm import tqdm
import pathlib

from utils import read_jsonl, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    actor_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        moral_action = sample["moral_action"]
        norm = sample["norm"]

        actor_data.append({
            "actor_input": f"{situation} {intention} {moral_action}",
            "actor_output": norm
        })

    datapath = pathlib.Path(args.datapath)
    write_json(actor_data, f"{datapath.parent}/actor_{datapath.stem}.json")

if __name__ == "__main__":
    main()