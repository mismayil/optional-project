import argparse
from tqdm import tqdm
import pathlib

from utils import (
    read_json, write_json, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN
)

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")

    args = parser.parse_args()

    data = read_json(args.datapath)
    critic_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        immoral_action = sample["immoral_action"]
        moral_action = sample["moral_action"]
        gold_norm = sample["norm"]
        anti_norm = sample["anti_norm"]
        other_norms = sample["other_norms"]
        other_anti_norms = sample["other_anti_norms"]
        context = f"{SITUATION_TOKEN} {situation} {INTENTION_TOKEN} {intention} {MORAL_ACTION_TOKEN} {moral_action} {IMMORAL_ACTION_TOKEN} {immoral_action} {NORM_TOKEN}"
        immoral_consequence = sample["immoral_consequence"]

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {gold_norm.lower()}",
            "critic_output": NO_HINT,
            "gold_norm": gold_norm
        }) 

        if anti_norm:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {anti_norm.lower()}",
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })

        for o_norm in other_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {o_norm.lower()}",
                "critic_output": NO_HINT,
                "gold_norm": gold_norm
            })

        for a_norm in other_anti_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {a_norm.lower()}",
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })
            
        fake_norms = sample["fake_norms"]

        for fake_norm in fake_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {fake_norm.lower()}",
                "critic_output": immoral_consequence.lower(),
                "gold_norm": gold_norm
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final{args.suffix}.json")

if __name__ == "__main__":
    main()