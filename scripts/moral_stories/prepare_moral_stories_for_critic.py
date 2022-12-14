import argparse
from tqdm import tqdm
import pathlib

from utils import read_json, write_json, get_phrases, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN

NO_HINT = "no hint"
JUGDMENT_HINT = "flip the judgment"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--concept-type", type=str, choices=["np", "vp"], default="vp", help="What type of concepts to extract from moral action and norms")

    args = parser.parse_args()

    data = read_json(args.datapath)
    critic_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        immoral_action = sample["immoral_action"]
        moral_action = sample["moral_action"]
        norm = sample["norm"]
        anti_norm = sample["anti_norm"]
        context = f"{SITUATION_TOKEN} {situation} {INTENTION_TOKEN} {intention} {MORAL_ACTION_TOKEN} {moral_action} {IMMORAL_ACTION_TOKEN} {immoral_action} {NORM_TOKEN}"
        norm_concepts = sample["norm_concepts"]
        moral_action_concepts = sample["moral_action_concepts"]

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {norm}",
            "critic_output": NO_HINT,
            "gold_norm": norm
        })

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {anti_norm}",
            "critic_output": JUGDMENT_HINT,
            "gold_norm": norm
        })

        all_concepts = set(list(map(str.lower, norm_concepts+moral_action_concepts)))

        for fake_norm in sample["fake_norms"]:
            fake_norm_concepts = get_phrases(fake_norm)
            
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {fake_norm}",
                "critic_output": ",".join(all_concepts-set(list(map(str.lower, fake_norm_concepts)))),
                "gold_norm": norm
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final.json")

if __name__ == "__main__":
    main()