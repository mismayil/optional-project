import argparse
from tqdm import tqdm
import pathlib
import spacy

from utils import read_json, write_json, get_noun_phrases

SEP_TOKEN = "<SEP>"
NO_HINT = "no hint"
JUGDMENT_HINT = "flip the judgment"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")

    args = parser.parse_args()

    data = read_json(args.datapath)
    critic_data = []

    nlp = spacy.load("en_core_web_sm")

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        immoral_action = sample["immoral_action"]
        moral_action = sample["moral_action"]
        norm = sample["norm"]
        anti_norm = sample["anti_norm"]
        context = f"{situation} {intention} {immoral_action}"
        norm_concepts = sample["norm_concepts"]
        moral_action_concepts = sample["moral_action_concepts"]

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {SEP_TOKEN} {norm}",
            "critic_output": NO_HINT,
            "gold_norm": norm,
            "moral_action": moral_action
        })

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {SEP_TOKEN} {anti_norm}",
            "critic_output": JUGDMENT_HINT,
            "gold_norm": norm,
            "moral_action": moral_action
        })

        all_concepts = set(list(map(str.lower, norm_concepts+moral_action_concepts)))

        for fake_norm in sample["fake_norms"]:
            fake_norm_doc = nlp(fake_norm)
            fake_norm_concepts = set()

            for token in fake_norm_doc:
                fake_norm_concepts.update(get_noun_phrases(token))
            
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {SEP_TOKEN} {fake_norm}",
                "critic_output": ",".join(all_concepts-set(list(map(str.lower, fake_norm_concepts)))),
                "gold_norm": norm,
                "moral_action": moral_action
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final.json")

if __name__ == "__main__":
    main()