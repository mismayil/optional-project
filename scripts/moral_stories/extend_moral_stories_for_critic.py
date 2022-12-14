import argparse
from tqdm import tqdm
import pathlib
import spacy

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-datapath", type=str, help="Path to critic dataset")
    parser.add_argument("--actor-datapath", type=str, help="Path to actor preds dataset")

    args = parser.parse_args()

    critic_data = read_json(args.critic_datapath)
    actor_data = read_json(args.actor_datapath)

    assert len(actor_data) == len(critic_data)

    for actor_sample, critic_sample in tqdm(zip(actor_data, critic_data), total=len(actor_data), desc="Preparing"):
        critic_sample["fake_norms"].append(actor_sample["prediction"])

    critic_datapath = pathlib.Path(args.critic_datapath)
    write_json(critic_data, f"{critic_datapath.parent}/{critic_datapath.stem}_extended.json")

if __name__ == "__main__":
    main()