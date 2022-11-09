import argparse
from tqdm import tqdm
import pathlib

from utils import read_json, write_json

SEP_TOKEN = "<SEP>"
NO_HINT = "No hint"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to glucose dataset")

    args = parser.parse_args()

    data = read_json(args.datapath)
    critic_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        context = " ".join(sample["context"])
        critic_data.append({
            "critic_input": f"{context} {SEP_TOKEN} {sample['actor_output']}",
            "critic_output": NO_HINT
        })

        for neg_output in sample["neg_actor_outputs"]:
            critic_data.append({
                "critic_input": f"{context} {SEP_TOKEN} {neg_output}",
                "critic_output": sample["critic_hint"]
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/critic_{datapath.stem}.json")

if __name__ == "__main__":
    main()