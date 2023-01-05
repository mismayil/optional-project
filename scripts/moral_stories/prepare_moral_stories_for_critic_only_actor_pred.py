import argparse
from tqdm import tqdm
import pathlib
from utils import (
    read_json, write_json, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN
)

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"
UNRELATED_HINT_1 = "unrelated 1:"
UNRELATED_HINT_2 = "unrelated 2:"
UNRELATED_HINT_3 = "unrelated 3:"
UNRELATED_HINT_4 = "unrelated 4:"

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
        context = f"{SITUATION_TOKEN} {situation} {INTENTION_TOKEN} {intention} {MORAL_ACTION_TOKEN} {moral_action} {IMMORAL_ACTION_TOKEN} {immoral_action} {NORM_TOKEN}"

        norm_concepts = list(map(str.lower, sample["norm_concepts"]))
        context_concepts = sample["context_concepts"]
        norm_action = sample["norm_action"] if sample["norm_action"] else gold_norm
        norm_sentiment = sample["norm_sentiment"]
        fake_norms = sample["fake_norms"]
        fake_norm_concepts = sample["fake_norm_concepts"]
        fake_norm_sentiments = sample["fake_norm_sentiments"]
        norm_grounded = (len(set(norm_concepts).intersection(set(context_concepts))) > 0)

        # only take actor prediction
        fake_norm = fake_norms[-1]
        fn_concepts = fake_norm_concepts[-1]
        fn_sentiment = fake_norm_sentiments[-1]

        fake_norm_grounded = (len(set(fn_concepts).intersection(set(context_concepts))) > 0)
        action_hint = norm_action.strip()
        
        if norm_sentiment * fn_sentiment < 0:
            if action_hint.lower().startswith("not"):
                action_hint = action_hint[3:].strip()
            else:
                action_hint = f"not {action_hint}"

        hint = f"{UNRELATED_HINT_4} {action_hint}"

        if norm_grounded and not fake_norm_grounded:
            concept_hint = ", ".join(set(norm_concepts)-set(list(map(str.lower, fn_concepts))))
            hint = f"{UNRELATED_HINT_1} {concept_hint}"
        
        elif not norm_grounded and fake_norm_grounded:
            hint = f"{UNRELATED_HINT_2} {action_hint}"
        
        elif norm_grounded and fake_norm_grounded:
            hint = f"{UNRELATED_HINT_3} {action_hint}"

        critic_data.append({
            "id": sample["id"],
            "actor_input": context,
            "actor_prediction": fake_norm.lower(),
            "critic_feedback": hint.lower(),
            "actor_reference": gold_norm
        })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final{args.suffix}.json")

if __name__ == "__main__":
    main()