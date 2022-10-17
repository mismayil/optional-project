import os
from torch.utils.data import random_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import pandas as pd

from glucose_t5_utils import GlucoseDataset

os.environ["WANDB_PROJECT"] = "optional-project"

if __name__ == "__main__":
    glucose_t5 = pd.read_csv("data/t5_training_data.tsv", sep="\t", header=None)
    glucose_t5_specific = pd.DataFrame()
    glucose_t5_specific["input"] = glucose_t5.iloc[:, 0]
    glucose_t5_specific["target"] = glucose_t5.iloc[:, 1].apply(lambda x: x.split("**")[0].strip())
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    dataset = GlucoseDataset(glucose_t5_specific, tokenizer)
    train_dataset, val_dataset = random_split(dataset, [len(dataset)-500, 500])
    training_args = Seq2SeqTrainingArguments(output_dir="/scratch/mete/glucose_t5_specific_out",
                                            logging_strategy="steps",
                                            num_train_epochs=1, 
                                            save_strategy="epoch",
                                            evaluation_strategy="steps",
                                            eval_steps=1000,
                                            logging_first_step=True, 
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