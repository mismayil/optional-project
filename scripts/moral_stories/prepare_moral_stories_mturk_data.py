import argparse
import re
import random
import pandas as pd
from tqdm import tqdm

from utils import read_json, MORAL_ACTION_TOKEN

CUSTOM_MORAL_ACTION_TOKEN = "| M_ACT | >"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to eval data.")
    args = parser.parse_args()

    data = read_json(args.datapath)
    independent_mturk_baseline_data = []
    independent_mturk_clue_data = []
    comparative_mturk_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        input_match = re.fullmatch("<\|SIT\|>(?P<situation>.*)<\|INT\|>(?P<intention>.*)<\|I_ACT\|>(?P<immoral_action>.*)<\|NRM\|>", sample["actor_input"])
        situation = input_match.group("situation").strip()
        intention = input_match.group("intention").strip()
        immoral_action = input_match.group("immoral_action").strip()
        actor_output = sample["actor_output"].split(MORAL_ACTION_TOKEN)
        actor_pred = sample["actor_pred"].split(CUSTOM_MORAL_ACTION_TOKEN)
        clue_actor_pred = sample["clue_actor_pred"].split(CUSTOM_MORAL_ACTION_TOKEN)
        gold_norm = actor_output[0].strip()
        baseline_norm = actor_pred[0].strip()
        clue_norm = clue_actor_pred[0].strip()
        coin = random.choice([0, 1])

        independent_mturk_baseline_data.append({
            "id": sample["id"],
            "situation": situation,
            "intention": intention,
            "immoral_action": immoral_action,
            "norm": baseline_norm,
            "gold_norm": gold_norm
        })

        independent_mturk_clue_data.append({
            "id": sample["id"],
            "situation": situation,
            "intention": intention,
            "immoral_action": immoral_action,
            "norm": clue_norm,
            "gold_norm": gold_norm
        })

        comparative_mturk_data.append({
            "id": sample["id"],
            "situation": situation,
            "intention": intention,
            "immoral_action": immoral_action,
            "norm1": baseline_norm if coin == 0 else clue_norm,
            "norm2": clue_norm if coin == 0 else baseline_norm,
            "gold_norm": gold_norm,
            "baseline_norm": 1 if coin == 0 else 2,
            "clue_norm": 2 if coin == 0 else 1,
        })
    
    pd.DataFrame(independent_mturk_baseline_data).to_csv("outputs/mturk_independent_baseline.csv", index=False)
    pd.DataFrame(independent_mturk_clue_data).to_csv("outputs/mturk_independent_clue.csv", index=False)
    pd.DataFrame(comparative_mturk_data).to_csv("outputs/mturk_comparative.csv", index=False)

if __name__ == "__main__":
    main()