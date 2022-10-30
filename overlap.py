from tqdm import tqdm
import pandas as pd
import re
import argparse
import pathlib

from utils import has_rule_overlap, has_story_overlap, has_story_copy

OVERLAP_THRESHOLD = 0.1
RULE_OVERLAP = "rule"
STORY_OVERLAP = "story"
STORY_COPY_OVERLAP = "copy"

def group_overlap_by_dim(data, dim=1, threshold=0.0):
    non_escaped = data[data[f"{dim}_specificNL"] != "escaped"]
    rule_overlap_col = f"{dim}_rule_overlap_{threshold}"
    story_overlap_col = f"{dim}_story_overlap_{threshold}"
    story_copy_overlap_col = f"{dim}_story_copy_overlap_{threshold}"
    group_cols = []

    if rule_overlap_col in data.columns:
        group_cols.append(rule_overlap_col)
    
    if story_overlap_col in data.columns:
        group_cols.append(story_overlap_col)

    if story_copy_overlap_col in data.columns:
        group_cols.append(story_copy_overlap_col)

    samples_by_overlap = non_escaped.groupby(group_cols).experiment_id.count()
    return samples_by_overlap.rename_axis(index={rule_overlap_col: "rule_overlap", story_overlap_col: "story_overlap", story_copy_overlap_col: "story_copy"}).rename(f"dim{dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD, help="Overlap threshold")
    parser.add_argument("--dims", type=int, nargs="+", default=list(range(1, 11)), help="Which dimensions to run the overlap for.")
    parser.add_argument("--dim-based-overlap", action="store_true", default=False, help="Whether to consider overlap with story based on dimension. e.g. for dim1 consider previous sentence")
    parser.add_argument("--overlap-types", type=str, nargs="+", choices=[RULE_OVERLAP, STORY_OVERLAP, STORY_COPY_OVERLAP], default=[RULE_OVERLAP, STORY_OVERLAP], help="What types of overlap to consider.")

    args = parser.parse_args()

    dim_rule_overlaps = [[], [], [], [], [], [], [], [], [], []]
    dim_story_overlaps = [[], [], [], [], [], [], [], [], [], []]
    dim_story_copy_overlaps = [[], [], [], [], [], [], [], [], [], []]

    data = pd.read_csv(args.dataset)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        for dim in args.dims:
            specific_rule = row[f"{dim}_specificNL"].strip()
            general_rule = row[f"{dim}_generalNL"].strip()

            if specific_rule == "escaped" or general_rule == "escaped":
                dim_rule_overlaps[dim-1].append([None, None])
                dim_story_overlaps[dim-1].append([None, None])
                dim_story_copy_overlaps[dim-1].append([None, None, None, None])
            else:
                specific_parts = re.split(">.*>", specific_rule)
                general_parts = re.split(">.*>", general_rule)

                assert len(specific_parts) == len(general_parts) == 2

                if RULE_OVERLAP in args.overlap_types:
                    rule_overlap = []

                    for sp_text, gl_text in zip(specific_parts, general_parts):
                        rule_overlap.append(has_rule_overlap(sp_text.strip(), gl_text.strip(), args.threshold))

                    dim_rule_overlaps[dim-1].append(rule_overlap)

                if STORY_OVERLAP in args.overlap_types:
                    specific_story_overlap = has_story_overlap(specific_parts[1].strip(), row['story'],
                                                                threshold=args.threshold,
                                                                dim_based_overlap=args.dim_based_overlap,
                                                                selected_context=row["selected_sentence"],
                                                                dim=dim)
                    general_story_overlap = has_story_overlap(general_parts[1].strip(), row['story'],
                                                                threshold=args.threshold,
                                                                dim_based_overlap=args.dim_based_overlap,
                                                                selected_context=row["selected_sentence"],
                                                                dim=dim)
                    dim_story_overlaps[dim-1].append([specific_story_overlap, general_story_overlap])
                
                if STORY_COPY_OVERLAP in args.overlap_types:
                    sp_copy_overlap, sp_copy_index, sp_copy_mode = has_story_copy(specific_parts[0].strip(), specific_parts[1].strip(), row["story"], threshold=args.threshold)
                    gl_copy_overlap, gl_copy_index, gl_copy_mode = has_story_copy(general_parts[0].strip(), general_parts[1].strip(), row["story"], threshold=args.threshold)
                    dim_story_copy_overlaps[dim-1].append([sp_copy_overlap, sp_copy_mode, gl_copy_overlap, gl_copy_mode])

    dim_overlap_df = pd.DataFrame()

    for dim in args.dims:
        if RULE_OVERLAP in args.overlap_types:
            dim_overlap_df[f"{dim}_rule_overlap_{args.threshold}"] = dim_rule_overlaps[dim-1]
        if STORY_OVERLAP in args.overlap_types:
            dim_overlap_df[f"{dim}_story_overlap_{args.threshold}"] = dim_story_overlaps[dim-1]
        if STORY_COPY_OVERLAP in args.overlap_types:
            dim_overlap_df[f"{dim}_story_copy_overlap_{args.threshold}"] = dim_story_copy_overlaps[dim-1]

    dims_text = "all" if len(args.dims) == 10 else "".join([str(d) for d in args.dims])
    dbo_text = "_dbo" if args.dim_based_overlap else ""
    overlap_type_text = "".join([otype[0] for otype in args.overlap_types])
    suffix = f"overlap_{args.threshold}_dim_{dims_text}_{overlap_type_text}{dbo_text}"

    data_with_overlap = pd.concat([data, dim_overlap_df], axis=1)
    with_overlap_file = f"outputs/{pathlib.Path(args.dataset).stem}_with_{suffix}.csv"
    data_with_overlap.to_csv(with_overlap_file)

    data_with_overlap = pd.read_csv(with_overlap_file)
    dim_overlap_stats_df = group_overlap_by_dim(data_with_overlap, dim=args.dims[0], threshold=args.threshold)

    for dim in args.dims[1:]:
        dim_overlap_stats_df = pd.merge(dim_overlap_stats_df, group_overlap_by_dim(data_with_overlap, dim=dim, threshold=args.threshold), left_index=True, right_index=True, how="outer")
    
    dim_overlap_stats_df = dim_overlap_stats_df.reset_index().fillna(0)
    dim_overlap_stats_df.to_csv(f"outputs/{pathlib.Path(args.dataset).stem}_stats_{suffix}.csv")