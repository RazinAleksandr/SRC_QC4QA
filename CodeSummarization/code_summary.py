import pandas as pd
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import argparse
from datasets import load_dataset

from data.DataPreprocessing import DataPreprocessing


def main(args):
    df = pd.read_csv('/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/sample_for_t5.csv')
    # dataset = load_dataset('RazinAleks/Python_SO_domains')
    # df = pd.concat(
    #     [
    #         # dataset["train"].to_pandas(),
    #         dataset["validation"].to_pandas(),
    #         # dataset["test"].to_pandas(),
    #     ],
    #     ignore_index=True,
    # )
    df.rename(columns={'Id_Q': 'Q_Id', 'Title_Q': 'Title', 'Body_Q': 'Question', 'Tags_Q': 'Tags'}, inplace=True)
    print(df.head(5))
    # Preprocess the data
    preprocessor = DataPreprocessing(df)
    preprocessor.preprocess_tags()
    preprocessor.extract_code_questions()

    # Generate code descriptions
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    descriptions = preprocessor.generate_code_descriptions(tokenizer, model, args.batch_size)

    # Substitute code snippets with descriptions
    result_df = preprocessor.substitute_code_with_description(descriptions)
    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--raw_data_path', type=str, required=True, help='Path to the raw data CSV file.')
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5-base-multi-sum', help='Name of the pre-trained model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for model inference.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output CSV file with code descriptions.')
    args = parser.parse_args()
    main(args)







