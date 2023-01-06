import argparse
from tqdm import tqdm
import pathlib
import re

from utils import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    critic_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        question = sample["question"]
        answer = sample["answer"]

        critic_data.append({
            "id": sample["id"],
            "critic_input": re.sub("\[QUERY\].*", "", re.sub("\[SITUATION\]", "", question)).strip().lower(),
            "critic_output": answer.strip().lower()
        })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/sc_critic_{datapath.stem}_final{args.suffix}.json")

if __name__ == "__main__":
    main()