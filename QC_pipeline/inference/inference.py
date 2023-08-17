import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import sys
import json
import time

sys.path.append('../models/transformers')

from E5_base import TextClassifier
from E5_small import SmallTextClassifier
from transformers import AutoTokenizer
from datasets import load_dataset

def gen_batch(df, batch_size):
    length = len(df)
    for i in range(0, length, batch_size):
        yield df.iloc[i : i + batch_size]

def timer(func):
    """A decorator that prints how long a function took to run."""
    
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        return result, elapsed_time

    return wrapper


def inference(df, classifier, tokenizer, threshold, batch_size):
    tags_classification = [
        "Data Science and Machine Learning",
        "Database and SQL",
        "GUI and Desktop Applications",
        "Networking and APIs",
        "Other",
        "Python Basics and Environment",
        "System Administration and DevOps",
        "Web Development",
    ]
    
    results = []
    batch_times = []

    classifier.eval()  # set the model to evaluation mode

    gen = gen_batch(df, batch_size)
    
    @timer
    def process_batch(batch):
        text_batch = (
            batch["Title"] + " " + batch["Question"]
        )  # combine 'Title' and 'Question'
        encoded_batch = tokenizer(
            text_batch.tolist(),
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():  # don't track gradients during the inference
            encoded_batch = {k: v.to(classifier.model.device) for k, v in encoded_batch.items()}
            output = classifier(encoded_batch)

        probs = F.sigmoid(output)  # get probabilities

        batch_results = {tag: (probs[:, idx] > threshold).int().tolist() for idx, tag in enumerate(tags_classification)}
        return pd.DataFrame(batch_results)

    for batch in tqdm(gen, total=int(len(df) / batch_size)):
        batch_result, elapsed_time = process_batch(batch)  # <-- using the decorated function
        results.append(batch_result)
        batch_times.append(elapsed_time)  # <-- append the time taken for the batch to the list

    results_df = pd.concat(results, ignore_index=True)
    df = pd.concat([df, results_df], axis=1)
    return df, batch_times  # <-- also return the batch times


def main(args):
    print(args)

    sys.path.append("/home/st-aleksandr-razin/workspace/SRC_QC4QA/models_zoo/classifiers/model.ckpt")

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
    classifier = TextClassifier("intfloat/e5-base", n_inputs=768, batch_norm=True)

    print("Loading model weights...")
    state_dict = torch.load(args.model_weights)
    state_dict = state_dict["state_dict"]
    new_state_dict = dict()

    for key, value in state_dict.items():
        if 't_' in key:
            new_state_dict[key[8:]] = value

    classifier.load_state_dict(new_state_dict)
    classifier.to(device)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    df = pd.concat(
        [
            # dataset["train"].to_pandas(),
            dataset["validation"].to_pandas(),
            # dataset["test"].to_pandas(),
        ],
        ignore_index=True,
    )
    print("Starting inference...")
    classified_df, batch_times = inference(df, classifier, tokenizer, args.threshold, args.batch_size)  # <-- capture the batch times here
    batch_times_df = pd.DataFrame({"batch_time_seconds": batch_times})
    print("Inference complete. Saving results...")
    batch_times_df.to_csv("batch_inference_times.csv", index=False)  # <-- save the batch times to a csv
    classified_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform inference on a given dataset with a trained model.')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for class assignment')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the inference')
    parser.add_argument('--device', type=str, default="cuda:0", help='Devide to run model')
    args = parser.parse_args()
    main(args)