import argparse
import os

from t5_experiments.eval.conala_eval import calculate_bleu_from_lists
from t5_experiments.t5_lm import T5LMClassifier
from t5_experiments.data_processing.utils import read_labels, get_encoded_code_tokens

DATA_FOLDER = 'data'

def training(training_file, dev_file,
             trained_models_dir,
             per_gpu_train_batch_size,
             learning_rate,
             epochs,
             language_model,
             grad_acc,
             sequence_length,
             optimizer_algorithm='adam',
             noisy_file=None,
             input_label='model_input',
             output_label='model_output',
             wandb_project="optional-project",
             wandb_run_name=None,
             tokenizer=None):

    if not os.path.exists(trained_models_dir):
        os.mkdir(trained_models_dir)
    classifier = T5LMClassifier(
            max_seq_length=sequence_length,
            output_model_dir=trained_models_dir,
            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
            pretrained_model_name_or_path=language_model,
            tokenizer_name_or_path=tokenizer
    )
    classifier.train(training_file, dev_file,
                    per_gpu_train_batch_size=per_gpu_train_batch_size,
                    learning_rate=learning_rate,
                    optimizer_algorithm=optimizer_algorithm,
                    num_train_epochs=epochs,
                    noisy_file=noisy_file,
                    gradient_accumulation_steps=grad_acc,
                    input_label=input_label,
                    output_label=output_label,
                    wandb_project=wandb_project,
                    wandb_run_name=wandb_run_name)


def evaluate(test_file, trained_models_dir, sequence_length,
             per_gpu_eval_batch_size, language_model, input_label="model_input", output_label="model_output", tokenizer=None):
    _classifier = T5LMClassifier(max_seq_length=sequence_length,
                                 output_model_dir=trained_models_dir,
                                 cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
                                 pretrained_model_name_or_path=language_model,
                                 tokenizer_name_or_path=tokenizer
                                 )

    preds = _classifier.predict(test_file=test_file,
                                per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                max_generated_tokens=sequence_length, input_label=input_label, output_label=output_label)
    labels = read_labels(test_file, tag=output_label)
    labels = [l.lower() for l in labels]
    preds = [p.lower() for p in preds]
    # labels = [' '.join(get_encoded_code_tokens(label)) for label in labels]
    eval_results = calculate_bleu_from_lists(gold_texts=labels, predicted_texts=preds)
    print(eval_results)
    return eval_results

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-file', dest='training_file', required=False, help='Path to training file',
                        default=None)
    parser.add_argument('--noisy-file', dest='noisy_file', required=False, help='Path to noisy file',
                        default=None)
    parser.add_argument('--validation-file', dest='validation_file', required=False, help='Path to validation file')
    parser.add_argument('--language-model', default='t5-base', help='Can be either some huggingface model or a '
                                                                         'path to a model. If the path is in GCS we '
                                                                         'download it first.')
    parser.add_argument('--tokenizer', type=str, default=None, help="Tokenizer for language model.")
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    parser.add_argument('--epochs', default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', default=128,
                        help='batch size')
    parser.add_argument('--val-batch-size', default=128,
                        help='validation batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='learning rate')
    parser.add_argument('--gradient-accumulation', default=1)
    parser.add_argument('--input-label', type=str, default="model_input", help="Input label name.")
    parser.add_argument('--output-label', type=str, default="model_output", help="Output label name.")
    parser.add_argument("--wandb-project", type=str, default="optional-project", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--max-seq-length", type=int, default=48, help="Maximum sequence length for training.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    #   train
    language_model = args.language_model
    if args.training_file and args.validation_file:
        training(training_file=args.training_file, dev_file=args.validation_file,
                 trained_models_dir=args.model_dir,
                 per_gpu_train_batch_size=int(args.batch_size),
                 epochs=int(args.epochs),
                 learning_rate=float(args.lr),
                 sequence_length=args.max_seq_length,
                 noisy_file=args.noisy_file,
                 language_model=language_model,
                 tokenizer=args.tokenizer,
                 grad_acc=int(args.gradient_accumulation),
                 input_label=args.input_label,
                 output_label=args.output_label,
                 wandb_project=args.wandb_project,
                 wandb_run_name=args.wandb_run_name)
    if args.validation_file:
            evaluation_results = evaluate(test_file=args.validation_file,
                                      trained_models_dir=args.model_dir,
                                      per_gpu_eval_batch_size=int(args.val_batch_size),
                                      sequence_length=args.max_seq_length,
                                      language_model=language_model,
                                      tokenizer=args.tokenizer,
                                      input_label=args.input_label,
                                      output_label=args.output_label
                                    )
if __name__ == '__main__':
    main()
