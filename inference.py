import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Path to T5 model")
    parser.add_argument("--dataset", type=str, help="Path to test dataset")

    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset, index_col=0, nrows=5)

    tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.model)

    predictions = []

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        input_ids = tokenizer(row["input"], return_tensors="pt", max_length=512, padding="max_length", truncation=True).input_ids
        outputs = model.generate(input_ids, max_length=512)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
    
    dataset["prediction"] = predictions

    dataset.to_csv(f"{pathlib.Path(args.dataset).stem}_with_preds.csv")