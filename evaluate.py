import pandas as pd
import sacrebleu
import argparse
import pathlib
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Path to the eval dataset")

    args = parser.parse_args()

    eval_data = pd.read_csv(args.dataset)

    scores = {"bleu": {f"dim{dim}": 0 for dim in range(1, 11)}}

    for dim in range(1, 11):
        dim_eval_data = eval_data[eval_data.dimension == dim]
        references = dim_eval_data.references.apply(lambda r: [s.strip().strip("'") for s in r.rstrip("]").strip("[").split(",")]).to_list()
        scores["bleu"][f"dim{dim}"] = sacrebleu.corpus_bleu(dim_eval_data.prediction.to_list(), references).score
    
    with open(f"{pathlib.Path(args.dataset).stem}_scores.json", "w") as f:
        json.dump(scores, f, indent=2)