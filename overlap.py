from tqdm import tqdm
import pandas as pd
import re
import argparse
import pathlib

from helpers import has_overlap, has_overlap_with_story

OVERLAP_THRESHOLD = 0.1

def group_overlap_by_dim(data, dim=1, threshold=0.0):
    non_escaped = data[data[f"{dim}_specificNL"] != "escaped"]
    spec_gen_col = f"{dim}_has_overlap_{threshold}"
    spec_story_col = f"{dim}_has_story_overlap_{threshold}"
    samples_by_overlap = non_escaped.groupby([spec_gen_col, spec_story_col]).experiment_id.count()
    return samples_by_overlap.rename_axis(index={spec_gen_col: "spec_gen_overlap", spec_story_col: "spec_story_overlap"}).rename(f"dim{dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD, help="Overlap threshold")

    args = parser.parse_args()

    dim_overlaps = [[], [], [], [], [], [], [], [], [], []]
    dim_story_overlaps = [[], [], [], [], [], [], [], [], [], []]

    data = pd.read_csv(args.dataset)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        for dim in range(1, 11):
            specific_rule = row[f"{dim}_specificNL"].strip()
            general_rule = row[f"{dim}_generalNL"].strip()

            if specific_rule == "escaped" or general_rule == "escaped":
                dim_overlaps[dim-1].append([None, None])
                dim_story_overlaps[dim-1].append(None)
            else:
                specific_parts = re.split(">.*>", specific_rule)
                general_parts = re.split(">.*>", general_rule)

                assert len(specific_parts) == len(general_parts) == 2

                overlap = []

                for sp_text, gl_text in zip(specific_parts, general_parts):
                    overlap.append(has_overlap(sp_text.strip(), gl_text.strip(), args.threshold))

                dim_overlaps[dim-1].append(overlap)

                story_overlap = has_overlap_with_story(specific_parts[1].strip(), row['story'], args.threshold)
                dim_story_overlaps[dim-1].append(story_overlap)

    dim_overlap_df = pd.DataFrame()

    for dim in range(1, 11):
        dim_overlap_df[f"{dim}_has_overlap_{args.threshold}"] = dim_overlaps[dim-1]
        dim_overlap_df[f"{dim}_has_story_overlap_{args.threshold}"] = dim_story_overlaps[dim-1]

    data_with_overlap = pd.concat([data, dim_overlap_df], axis=1)
    data_with_overlap.to_csv(f"{pathlib.Path(args.dataset).stem}_with_overlap_{args.threshold}.csv")

    dim_overlap_stats_df = group_overlap_by_dim(data_with_overlap, dim=1, threshold=args.threshold)

    for dim in range(2, 11):
        dim_overlap_stats_df = pd.merge(dim_overlap_stats_df, group_overlap_by_dim(data_with_overlap, dim=dim, threshold=args.threshold), left_index=True, right_index=True, how="outer")
    
    dim_overlap_stats_df = dim_overlap_stats_df.reset_index().fillna(0)
    dim_overlap_stats_df.to_csv(f"{pathlib.Path(args.dataset).stem}_overlap_{args.threshold}_stats.csv")