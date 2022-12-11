import argparse
from tqdm import tqdm
import pathlib

from utils import read_jsonl, write_json

SITUATION_TOKEN = "<|SIT|>"
INTENTION_TOKEN = "<|INT|>"
MORAL_ACTION_TOKEN = "<|M_ACT|>"
IMMORAL_ACTION_TOKEN = "<|I_ACT|>"
NORM_TOKEN = "<|NRM|>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    actor_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        immoral_action = sample["immoral_action"]
        moral_action = sample["moral_action"]
        norm = sample["norm"]

        actor_data.append({
            "actor_input": f"{SITUATION_TOKEN} {situation} {INTENTION_TOKEN} {intention} {MORAL_ACTION_TOKEN} {moral_action} {IMMORAL_ACTION_TOKEN} {immoral_action} {NORM_TOKEN}",
            "actor_output": norm
        })

    datapath = pathlib.Path(args.datapath)
    write_json(actor_data, f"{datapath.parent}/actor_{datapath.stem}.json")

if __name__ == "__main__":
    main()