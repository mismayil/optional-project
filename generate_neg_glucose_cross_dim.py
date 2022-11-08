import argparse
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pathlib
from tqdm import tqdm

from utils import read_json, write_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_TOKEN = "<MASK>"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_negative_context(sample):
    mask_index = sample["context"].index(MASK_TOKEN)

    if sample["actor_dim"] == 1:
        negative_context = sample["context"][:mask_index]
        negative_context += [sample["masked_sentence"]] if sample["masked_sentence"] is not None else []
        negative_sent_idx = np.random.choice(list(range(mask_index+1, len(sample["context"]))))
        negative_context += [sent.strip("*") for sent in sample["context"][mask_index+1:negative_sent_idx]]
        negative_context += [f"*{sample['context'][negative_sent_idx].strip('*')}*"]
        negative_context += [MASK_TOKEN]
        negative_context += [sent.strip("*") for sent in sample["context"][negative_sent_idx+2:len(sample["context"])]]
        return negative_context
    
    negative_sent_idx = np.random.choice(list(range(mask_index)))
    negative_context = sample["context"][:max(0, negative_sent_idx-1)] + [MASK_TOKEN]
    negative_context += [f"*{sample['context'][negative_sent_idx].strip('*')}*"]
    negative_context += [sent.strip("*") for sent in sample["context"][negative_sent_idx+1:mask_index]]
    negative_context += [sample["masked_sentence"]] if sample["masked_sentence"] is not None else []
    negative_context += [sent.strip("*") for sent in sample["context"][mask_index+1:len(sample["context"])]]
    return negative_context

def generate_negative_samples(data, model, tokenizer, batch_size=32, max_input_length=256, max_output_length=60, num_generate=3):
    for batch in tqdm(chunks(data, batch_size), total=(len(data)/batch_size), desc="Generating"):
        tokenized_data = tokenizer.batch_encode_plus(
                    [" ".join(generate_negative_context(sample)) for sample in batch],
                    max_length=max_input_length,
                    padding='longest',
                    return_tensors="pt",
                )
        outputs = model.generate(input_ids=tokenized_data.input_ids.to(device),
                                 attention_mask=tokenized_data.attention_mask.to(device),
                                 max_new_tokens=max_output_length, temperature=0.7,
                                 top_p=0.9, num_return_sequences=num_generate, num_beams=3)
        generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations = [generations[i:i+num_generate] for i in range(0, len(generations), num_generate)]

        for sample, gens in zip(batch, generations):
            sample["neg_actor_outputs"] = gens

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to glucose dataset")
    parser.add_argument("--tokenizer", type=str, default="allenai/unifiedqa-t5-base", help="Tokenizer path")
    parser.add_argument("--model", type=str, default="/scratch/mete/op_baseline1_actor_glucose_cross_dim", help="Model path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-input-length", type=int, default=256, help="Max input sequence length")
    parser.add_argument("--max-output-length", type=int, default=60, help="Max output sequence length")
    parser.add_argument("--num-sequences", type=int, default=3, help="Number of sequences to generate per sample")

    args = parser.parse_args()

    data = read_json(args.datapath)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    data = generate_negative_samples(data, model, tokenizer,
                                    batch_size=args.batch_size, 
                                    max_input_length=args.max_input_length, 
                                    max_output_length=args.max_output_length,
                                    num_generate=args.num_sequences)
    datapath = pathlib.Path(args.datapath)
    write_json(data, f"{datapath.parent}/{datapath.stem}_with_neg.json")

if __name__ == "__main__":
    main()