import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import pathlib
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlucoseCrossDimDataset(Dataset):
    def __init__(self, data, tokenizer, input_label="model_input", output_label="model_output", max_input_len=512, max_output_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.input_label = input_label
        self.output_label = output_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.tokenizer(
            [self.data[index][self.input_label]],
            padding="max_length",
            max_length=self.max_input_len,
            return_tensors="pt",
            truncation=True,
        )
        target = self.tokenizer(
            [self.data[index][self.output_label]],
            padding="max_length",
            max_length=self.max_output_len,
            return_tensors="pt",
            truncation=True,
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "labels": target_ids.to(dtype=torch.long),
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Path to T5 model")
    parser.add_argument("--tokenizer", type=str, help="Path to T5 Tokenizer")
    parser.add_argument("--dataset", type=str, help="Path to test dataset")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output file.")
    parser.add_argument("--max-length", type=int, default=48, help="Maximum sequence length.")
    parser.add_argument("--input-label", type=str, default="model_input", help="Input label.")
    parser.add_argument("--output-label", type=str, default="model_output", help="Output label.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")

    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.to(device)

    eval_dataset = []

    for data in tqdm(dataset, total=len(dataset)):
        input_ids = tokenizer(data[args.input_label], return_tensors="pt", max_length=args.max_length, padding="max_length", truncation=True).input_ids
        outputs = model.generate(input_ids.to(device), max_length=args.max_length)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        references = [data[args.output_label]] if isinstance(data[args.output_label], str) else data[args.output_label]
        eval_dataset.append({"references": references, "prediction": prediction})

    with open(f"outputs/{pathlib.Path(args.dataset).stem}_with_preds{args.suffix}.csv", "w") as f:
        json.dump(eval_dataset, f, indent=2)