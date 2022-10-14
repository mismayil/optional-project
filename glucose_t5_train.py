import os
import torch
from torch.utils.data import Dataset, random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import pandas as pd

os.environ["WANDB_PROJECT"] = "optional-project"

class GlucoseDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_len=512, max_target_len=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.tokenizer(
            [self.data.iloc[index, 0]],
            pad_to_max_length=True,
            max_length=self.max_source_len,
            return_tensors="pt",
            truncation=True,
        )
        target = self.tokenizer(
            [self.data.iloc[index, 1]],
            pad_to_max_length=True,
            max_length=self.max_target_len,
            return_tensors="pt",
            truncation=True,
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "decoder_input_ids": target_ids.to(dtype=torch.long),
            "decoder_attention_mask": target_mask.to(dtype=torch.long),
        }

if __name__ == "__main__":
    glucose = pd.read_csv("data/glucose_t5_specific.tsv", sep="\t", header=None)
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    dataset = GlucoseDataset(glucose, tokenizer)
    train_dataset, val_dataset = random_split(dataset, [500, len(dataset)-500])
    training_args = Seq2SeqTrainingArguments(output_dir="t5_outputs",
                                            logging_strategy="epoch",
                                            num_train_epochs=5, 
                                            save_strategy="epoch", 
                                            report_to="wandb",
                                            run_name="glucose_t5_specific",
                                            per_device_train_batch_size=2,
                                            per_device_eval_batch_size=2)
    trainer = Seq2SeqTrainer(model=model,
                            args=training_args,
                            data_collator=DataCollatorForSeq2Seq(tokenizer, model),
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset)
    trainer.train()