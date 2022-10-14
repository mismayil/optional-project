import os
from torch.utils.data import random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import pandas as pd

from glucose_t5_utils import GlucoseDataset

os.environ["WANDB_PROJECT"] = "optional-project"

if __name__ == "__main__":
    glucose = pd.read_csv("data/glucose_t5_general.tsv", sep="\t", header=None)
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    dataset = GlucoseDataset(glucose, tokenizer)
    train_dataset, val_dataset = random_split(dataset, [len(dataset)-500, 500])
    training_args = Seq2SeqTrainingArguments(output_dir="glucose_t5_general_out",
                                            logging_strategy="epoch",
                                            num_train_epochs=5, 
                                            save_strategy="epoch",
                                            evaluation_strategy="epoch", 
                                            report_to="wandb",
                                            run_name="glucose_t5_general",
                                            per_device_train_batch_size=4,
                                            per_device_eval_batch_size=4)
    trainer = Seq2SeqTrainer(model=model,
                            args=training_args,
                            data_collator=DataCollatorForSeq2Seq(tokenizer, model),
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset)
    trainer.train()