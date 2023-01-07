import argparse
from tqdm import tqdm
import pathlib
import random
import math

from utils import (
    read_json, write_json, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN,
    GOOD_NORM_PREFIXES, BAD_NORM_PREFIXES, SYN_ANT_MAP
)

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"
UNRELATED_HINT_1 = "unrelated 1:"
UNRELATED_HINT_2 = "unrelated 2:"
UNRELATED_HINT_3 = "unrelated 3:"
UNRELATED_HINT_4 = "unrelated 4:"

def replicate_norm(norm, sentiment=1, num=1):
    def _find_prefix_index(prefixes):
        for i, prefix_lst in enumerate(prefixes):
            for prefix in prefix_lst:
                if prefix.lower() in norm.lower():
                    return i, prefix
        return -1, None

    prefixes = GOOD_NORM_PREFIXES

    if sentiment == -1:
        prefixes = BAD_NORM_PREFIXES
    
    exc_index, exc_prefix = _find_prefix_index(prefixes)

    if exc_index != -1:
        candidate_indices = random.sample(list(set(range(len(prefixes))).difference(set([exc_index]))), min(len(prefixes)-1, num))
        return [norm.lower().replace(exc_prefix.lower(), random.choice(prefixes[idx])) for idx in candidate_indices]
    
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")
    parser.add_argument("--balanced", action="store_true", default=False, help="Whether to make dataset balanced")

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

        critic_data.append({
            "id": sample["id"],
            "critic_input": f"{context} {gold_norm.lower()}",
            "critic_output": NO_HINT,
            "gold_norm": gold_norm
        }) 

        if anti_norm:
            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {anti_norm.lower()}",
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })

        norm_concepts = list(map(str.lower, sample["norm_concepts"]))
        context_concepts = sample["context_concepts"]
        norm_action = sample["norm_action"] if sample["norm_action"] else gold_norm
        norm_sentiment = sample["norm_sentiment"]
        fake_norms = sample["fake_norms"]
        fake_norm_concepts = sample["fake_norm_concepts"]
        fake_norm_sentiments = sample["fake_norm_sentiments"]
        norm_grounded = (len(set(norm_concepts).intersection(set(context_concepts))) > 0)

        type_1_cases = []
        type_2_cases = []
        type_3_cases = []
        type_4_cases = []

        for fake_norm, fn_concepts, fn_sentiment in zip(fake_norms, fake_norm_concepts, fake_norm_sentiments):
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
                type_1_cases.append({"norm": fake_norm, "hint": hint, "sentiment": fn_sentiment})
            
            elif not norm_grounded and fake_norm_grounded:
                hint = f"{UNRELATED_HINT_2} {action_hint}"
                type_2_cases.append({"norm": fake_norm, "hint": hint, "sentiment": fn_sentiment})
            
            elif norm_grounded and fake_norm_grounded:
                hint = f"{UNRELATED_HINT_3} {action_hint}"
                type_3_cases.append({"norm": fake_norm, "hint": hint, "sentiment": fn_sentiment})

            else:
                type_4_cases.append({"norm": fake_norm, "hint": hint, "sentiment": fn_sentiment})

            critic_data.append({
                "id": sample["id"],
                "critic_input": f"{context} {fake_norm.lower()}",
                "critic_output": hint.lower(),
                "gold_norm": gold_norm
            })

        if args.balanced:
            max_error_cases = max(1, max(len(type_1_cases), len(type_2_cases), len(type_3_cases), len(type_4_cases)))

            extra_other_norms = []
            extra_other_anti_norms = []
            max_no_hint_cases = max_error_cases-1
            max_contr_cases = max_error_cases-1

            if len(other_norms) >= max_no_hint_cases:
                extra_other_norms = random.sample(other_norms, max_no_hint_cases)
            else:
                extra_other_norms = replicate_norm(gold_norm, norm_sentiment, max_no_hint_cases)

            if len(other_anti_norms) >= max_contr_cases:
                extra_other_anti_norms = random.sample(other_anti_norms, max_contr_cases)
            else:
                if anti_norm:
                    extra_other_anti_norms = replicate_norm(anti_norm, norm_sentiment, max_contr_cases)
            
            for o_norm in extra_other_norms:
                critic_data.append({
                    "id": sample["id"],
                    "critic_input": f"{context} {o_norm.lower()}",
                    "critic_output": NO_HINT,
                    "gold_norm": gold_norm
                })

            for a_norm in extra_other_anti_norms:
                critic_data.append({
                    "id": sample["id"],
                    "critic_input": f"{context} {a_norm.lower()}",
                    "critic_output": CONTRADICTION_HINT,
                    "gold_norm": gold_norm
                }) 

            extra_type_1_length = max_error_cases-len(type_1_cases)
            extra_type_2_length = max_error_cases-len(type_2_cases)
            extra_type_3_length = max_error_cases-len(type_3_cases)
            extra_type_4_length = max_error_cases-len(type_4_cases)
            error_cases = [type_1_cases[:extra_type_1_length], type_2_cases[:extra_type_2_length], type_3_cases[:extra_type_3_length], type_4_cases[:extra_type_4_length]]
            
            def _get_ratio(cases, length):
                if len(cases) == 0:
                    return 0
                
                return math.ceil(length/len(cases))

            error_ratios = [_get_ratio(type_1_cases, extra_type_1_length), _get_ratio(type_2_cases, extra_type_2_length), _get_ratio(type_3_cases, extra_type_3_length), _get_ratio(type_4_cases, extra_type_4_length)]
            
            for ec_lst, ec_ratio in zip(error_cases, error_ratios):
                for ec in ec_lst:
                    e_norms = replicate_norm(ec["norm"], sentiment=ec["sentiment"], num=ec_ratio)
                    if e_norms:
                        for e_norm in e_norms:
                                critic_data.append({
                                    "id": sample["id"],
                                    "critic_input": f"{context} {e_norm.lower()}",
                                    "critic_output": ec["hint"].lower(),
                                    "gold_norm": gold_norm
                                })
        else:
            for o_norm in other_norms:
                critic_data.append({
                    "id": sample["id"],
                    "critic_input": f"{context} {o_norm.lower()}",
                    "critic_output": NO_HINT,
                    "gold_norm": gold_norm
                })

            for a_norm in other_anti_norms:
                critic_data.append({
                    "id": sample["id"],
                    "critic_input": f"{context} {a_norm.lower()}",
                    "critic_output": CONTRADICTION_HINT,
                    "gold_norm": gold_norm
                })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final{args.suffix}.json")

if __name__ == "__main__":
    main()