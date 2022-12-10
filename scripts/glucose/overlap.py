from tqdm import tqdm
import pandas as pd
import re
import argparse
import pathlib

from utils import has_overlap, has_story_overlap

OVERLAP_THRESHOLD = 0.1
RULE_OVERLAP = "rule"
STORY_OVERLAP = "story"

def group_overlap_by_dim(data, dim=1, threshold=0.0):
    non_escaped = data[data[f"{dim}_specificNL"] != "escaped"]
    rule_head_overlap_col = f"{dim}_rule_head_overlap_{threshold}"
    rule_tail_overlap_col = f"{dim}_rule_tail_overlap_{threshold}"
    story_spec_overlap_col = f"{dim}_story_spec_overlap_{threshold}"
    story_gen_overlap_col = f"{dim}_story_gen_overlap_{threshold}"
    group_cols = []

    if rule_head_overlap_col in data.columns:
        group_cols.append(rule_head_overlap_col)
        group_cols.append(rule_tail_overlap_col)
    
    if story_spec_overlap_col in data.columns:
        group_cols.append(story_spec_overlap_col)
        group_cols.append(story_gen_overlap_col)

    samples_by_overlap = non_escaped.groupby(group_cols).experiment_id.count()
    return samples_by_overlap.rename_axis(index={rule_head_overlap_col: "rule_head_overlap", 
                                                 rule_tail_overlap_col: "rule_tail_overlap", 
                                                 story_spec_overlap_col: "story_spec_overlap",
                                                 story_gen_overlap_col: "story_gen_overlap",}).rename(f"dim{dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD, help="Overlap threshold")
    parser.add_argument("--dims", type=int, nargs="+", default=list(range(1, 11)), help="Which dimensions to run the overlap for.")
    parser.add_argument("--overlap-types", type=str, nargs="+", choices=[RULE_OVERLAP, STORY_OVERLAP], default=[RULE_OVERLAP, STORY_OVERLAP], help="What types of overlap to consider.")
    parser.add_argument("--ignore-stopwords", action="store_true", default=False, help="Whether to ignore stopwords when computing overlap.")
    parser.add_argument("--ignore-entities", action="store_true", default=False, help="Whether to ignore named entities when computing overlap.")

    args = parser.parse_args()

    dim_rule_overlaps = [[], [], [], [], [], [], [], [], [], []]
    dim_story_overlaps = [[], [], [], [], [], [], [], [], [], []]

    data = pd.read_csv(args.dataset)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        for dim in args.dims:
            specific_rule = row[f"{dim}_specificNL"].strip()
            general_rule = row[f"{dim}_generalNL"].strip()

            if specific_rule == "escaped" or general_rule == "escaped":
                dim_rule_overlaps[dim-1].append([None, None])
                dim_story_overlaps[dim-1].append([None, None])
            else:
                specific_parts = re.split(">.*>", specific_rule)
                general_parts = re.split(">.*>", general_rule)

                assert len(specific_parts) == len(general_parts) == 2

                if RULE_OVERLAP in args.overlap_types:
                    rule_overlap = []

                    for sp_text, gl_text in zip(specific_parts, general_parts):
                        rule_overlap.append(has_overlap(sp_text.strip(), gl_text.strip(), args.threshold, 
                                                        ignore_stopwords=args.ignore_stopwords,
                                                        ignore_entities=args.ignore_entities))

                    dim_rule_overlaps[dim-1].append(rule_overlap)

                if STORY_OVERLAP in args.overlap_types:
                    specific_story_overlap = has_story_overlap(specific_parts[0].strip(), specific_parts[1].strip(),
                                                                row['story'],
                                                                threshold=args.threshold,
                                                                selected_context=row["selected_sentence"],
                                                                dim=dim,
                                                                ignore_stopwords=args.ignore_stopwords,
                                                                ignore_entities=args.ignore_entities)
                    general_story_overlap = has_story_overlap(general_parts[0].strip(), general_parts[1].strip(),
                                                                row['story'],
                                                                threshold=args.threshold,
                                                                selected_context=row["selected_sentence"],
                                                                dim=dim,
                                                                ignore_stopwords=args.ignore_stopwords,
                                                                ignore_entities=args.ignore_entities)
                    dim_story_overlaps[dim-1].append([specific_story_overlap, general_story_overlap])

    dim_overlap_df = pd.DataFrame()

    for dim in args.dims:
        if RULE_OVERLAP in args.overlap_types:
            dim_overlap_df[f"{dim}_rule_head_overlap_{args.threshold}"] = [o[0] for o in dim_rule_overlaps[dim-1]]
            dim_overlap_df[f"{dim}_rule_tail_overlap_{args.threshold}"] = [o[1] for o in dim_rule_overlaps[dim-1]]
        if STORY_OVERLAP in args.overlap_types:
            dim_overlap_df[f"{dim}_story_spec_overlap_{args.threshold}"] = [o[0] for o in dim_story_overlaps[dim-1]]
            dim_overlap_df[f"{dim}_story_gen_overlap_{args.threshold}"] = [o[1] for o in dim_story_overlaps[dim-1]]

    dims_text = "all" if len(args.dims) == 10 else "".join([str(d) for d in args.dims])
    overlap_type_text = "".join([otype[0] for otype in args.overlap_types])
    suffix = f"overlap_{args.threshold}_dim_{dims_text}_{overlap_type_text}"

    data_with_overlap = pd.concat([data, dim_overlap_df], axis=1)
    with_overlap_file = f"large_outputs/{pathlib.Path(args.dataset).stem}_with_{suffix}.csv"
    data_with_overlap.to_csv(with_overlap_file)

    data_with_overlap = pd.read_csv(with_overlap_file)
    dim_overlap_stats_df = group_overlap_by_dim(data_with_overlap, dim=args.dims[0], threshold=args.threshold)

    for dim in args.dims[1:]:
        dim_overlap_stats_df = pd.merge(dim_overlap_stats_df, group_overlap_by_dim(data_with_overlap, dim=dim, threshold=args.threshold), left_index=True, right_index=True, how="outer")
    
    dim_overlap_stats_df = dim_overlap_stats_df.reset_index().fillna(0)
    dim_overlap_stats_df.to_csv(f"outputs/{pathlib.Path(args.dataset).stem}_stats_{suffix}.csv")