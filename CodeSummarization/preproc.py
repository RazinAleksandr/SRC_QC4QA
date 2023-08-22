import pandas as pd
from tqdm import tqdm
import random

from transformers import RobertaTokenizer, T5ForConditionalGeneration

from data.DataPreprocessing import DataPreprocessing


def main():
    path_raw = '/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/summarization/raw/python_QA_text_code_raw.csv'
    path_code = '/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/summarization/processed/code_question_df.csv'
    path_described_code = '/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/summarization/processed/code_question_described_df.csv'
    question_with_code_description_path = '/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/summarization/final/question_with_code_description.csv'
    question_with_code_description_path_raw = '/home/st-aleksandr-razin/workspace/SRC_QC4QA/data/summarization/final/question_with_code_description_raw.csv'


    df = pd.read_csv(path_raw, index_col=0)
    preprocessing = DataPreprocessing(df)
    preprocessing.preprocess_tags()
    preprocessing.indexing_code()
    
    code_question_df = preprocessing.get_code_question_dataframe()
    code_question_df.to_csv(path_code, index=False)

    # Randomly sample 50,000 unique question_ids
    sampled_question_ids = random.sample(code_question_df['question_id'].unique().tolist(), 50000)

    # Filter the code_df based on the sampled question_ids
    filtered_code_df = code_question_df[code_question_df['question_id'].isin(sampled_question_ids)]
    filtered_code_df.reset_index(inplace=True, drop=True)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    described_df = preprocessing.generate_code_descriptions(tokenizer, model, filtered_code_df, batch_size=96)
    described_df.to_csv(path_described_code, index=False)

    question_with_code_description, raw_df = preprocessing.substitute_code_with_description(described_df)
    question_with_code_description = question_with_code_description[question_with_code_description['Id_Q'].isin(sampled_question_ids)]
    raw_df = raw_df[raw_df['Id_Q'].isin(sampled_question_ids)]
    
    raw_df.reset_index(drop=True, inplace=True)
    question_with_code_description.reset_index(drop=True, inplace=True)
    l = []
    i = 0
    for j, code in tqdm(enumerate(question_with_code_description['Body_Q'].values)):
        if 'code description start:' in code and '<pre><code>' not in code: i += 1
        else: 
            l.append(j)
    question_with_code_description.drop(l, inplace=True)
    raw_df.drop(l, inplace=True)
    
    raw_df.reset_index(drop=True, inplace=True)
    question_with_code_description.reset_index(drop=True, inplace=True)


    question_with_code_description.to_csv(question_with_code_description_path, index=False)
    raw_df.to_csv(question_with_code_description_path_raw, index=False)


if __name__ == "__main__":
    main()