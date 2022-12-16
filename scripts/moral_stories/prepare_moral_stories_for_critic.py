import argparse
from tqdm import tqdm
import pathlib

from utils import read_json, write_json, get_phrases, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"
UNRELATED_HINT_1 = "unrelated 1:"
UNRELATED_HINT_2 = "unrelated 2:"
UNRELATED_HINT_3 = "unrelated 3:"
UNRELATED_HINT_4 = "unrelated 4:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")

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

        for norm in [gold_norm] + other_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {norm}",
                "critic_output": NO_HINT,
                "gold_norm": gold_norm
            }) 

        for norm in [anti_norm] + other_anti_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {norm}",
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })

        norm_concepts = list(map(str.lower, sample["norm_concepts"]))
        context_concepts = sample["context_concepts"]
        norm_action = sample["norm_action"] if sample["norm_action"] else gold_norm
        fake_norms = sample["fake_norms"]
        fake_norm_concepts = sample["fake_norm_concepts"]
        norm_grounded = (len(set(norm_concepts).intersection(set(context_concepts))) > 0)

        for fake_norm, fn_concepts in zip(fake_norms, fake_norm_concepts):
            fake_norm_grounded = (len(set(fn_concepts).intersection(set(context_concepts))) > 0)
            hint = f"{UNRELATED_HINT_4} {norm_action.strip()}"

            if norm_grounded and not fake_norm_grounded:
                feedback = ",".join(set(norm_concepts)-set(list(map(str.lower, fn_concepts))))
                hint = f"{UNRELATED_HINT_1} {feedback}"
            
            if not norm_grounded and fake_norm_grounded:
                hint = f"{UNRELATED_HINT_2} {norm_action.strip()}"
            
            if norm_grounded and fake_norm_grounded:
                hint = f"{UNRELATED_HINT_3} {norm_action.strip()}"

            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {fake_norm}",
                "critic_output": hint,
                "gold_norm": gold_norm
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final.json")

if __name__ == "__main__":
    main()