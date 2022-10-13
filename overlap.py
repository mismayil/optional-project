from tqdm import tqdm
import pandas as pd
import re

from helpers import has_overlap, has_overlap_with_story

OVERLAP_THRESHOLD = 0.9
dim_overlaps = [[], [], [], [], [], [], [], [], [], []]
dim_story_overlaps = [[], [], [], [], [], [], [], [], [], []]

glucose = pd.read_csv("GLUCOSE_training_data_final.csv")

for i, row in tqdm(glucose.iterrows(), total=len(glucose)):
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
                overlap.append(has_overlap(sp_text.strip(), gl_text.strip(), OVERLAP_THRESHOLD))

            dim_overlaps[dim-1].append(overlap)

            story_overlap = has_overlap_with_story(specific_parts[1].strip(), row['story'], OVERLAP_THRESHOLD)
            dim_story_overlaps[dim-1].append(story_overlap)

dim_overlap_df = pd.DataFrame()

for dim in range(1, 11):
    dim_overlap_df[f"{dim}_has_overlap_{OVERLAP_THRESHOLD}"] = dim_overlaps[dim-1]
    dim_overlap_df[f"{dim}_has_story_overlap_{OVERLAP_THRESHOLD}"] = dim_story_overlaps[dim-1]

dim_overlap_df.to_csv("glucose_with_overlap.csv")