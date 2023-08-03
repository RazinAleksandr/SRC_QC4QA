import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import random

from transformers import RobertaTokenizer, T5ForConditionalGeneration, DataCollatorWithPadding


class DataPreprocessing:
    def __init__(self, df=None, columns=['Id_Q', 'Title_Q', 'Body_Q', 'Tags_Q', 'Code_Q']):
        self.df = df[columns].drop_duplicates()
    
    def preprocess_tags(self):
        self.df['Tags_Q'] = self.df['Tags_Q'].apply(lambda x: x.split(','))
    
    def indexing_code(self):
        self.code_df = self.df[self.df['Code_Q']][['Id_Q']]
    
    def get_code_question_dataframe(self):
        code_question_data = []
        start_tag = '<pre><code>'
        end_tag = '</code></pre>'
        
        for index, row in tqdm(self.df.iterrows()):
            if row['Id_Q'] in self.code_df['Id_Q'].values and pd.notnull(row['Body_Q']):
                text = row['Body_Q']
                start_index = text.find(start_tag)
                pos_code = 0
                while start_index != -1:
                    end_index = text.find(end_tag, start_index)
                    
                    if end_index != -1:
                        code_snippet = text[start_index + len(start_tag):end_index]
                        code_question_data.append({'code_id': pos_code, 'code_snippet': code_snippet,'question_id': row['Id_Q']})
                        pos_code += 1
                        start_index = text.find(start_tag, end_index)
                    else:
                        break
        code_question_df = pd.DataFrame(code_question_data)
        return code_question_df
    
    def generate_code_descriptions(self, tokenizer, model, code_question_df, batch_size=64):
        code_question_df['code_description'] = ''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dataset = CodeDataset(code_question_df, tokenizer)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer, padding=True))

        for batch in tqdm(dataloader):
            #batch_indices = batch_indices.tolist()

            with torch.no_grad():
                generated_ids = model.generate(batch["input_ids"].to(device), max_length=20)

            descriptions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, description in enumerate(descriptions):
                index = int(batch['index'][i])
                code_question_df.loc[index, 'code_description'] = str(description)

        return code_question_df

    def substitute_code_with_description(self, code_df):
        for row in tqdm(code_df.itertuples()):
            code_id = row.code_id
            question_id = row.question_id
            description = row.code_description
            code_snip = row.code_snippet
            
            # Find the corresponding question based on question ID
            question = self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'].values[0]
            
            # Replace the code with its description in the question
            question = question.replace(f'<pre><code>{code_snip}</code></pre>', f'code description start: {description} code description end')
            
            # Update the question in the question DataFrame
            self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'] = question
        
        return self.df

class CodeDataset(Dataset):
    def __init__(self, code_question_df, tokenizer):
        self.code_snippets = code_question_df['code_snippet'].tolist()
        self.indices = torch.arange(len(self.code_snippets))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, index):
        code_snippet = self.code_snippets[index]
        encoded_input = self.tokenizer.encode_plus(
            code_snippet,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded_input["input_ids"]
        return {"input_ids": input_ids[0], "index": index}


def main():
    path_raw = '/home/arazin/main/work/HUAWEI/SRC/QCQA/question_classification/dataset/raw_data/python_QA_text_code_raw.csv'
    path_code = '/home/arazin/main/work/HUAWEI/SRC/QCQA/question_classification/dataset/processed_data/code_question_df.csv'
    path_described_code = '/home/arazin/main/work/HUAWEI/SRC/QCQA/question_classification/dataset/processed_data/code_question_described_df.csv'
    question_with_code_description_path = '/home/arazin/main/work/HUAWEI/SRC/QCQA/question_classification/dataset/processed_data/question_with_code_description.csv'

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

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    described_df = preprocessing.generate_code_descriptions(tokenizer, model, filtered_code_df, batch_size=96)
    described_df.to_csv(path_described_code, index=False)

    question_with_code_description = preprocessing.substitute_code_with_description(described_df)
    question_with_code_description = question_with_code_description[question_with_code_description['Id_Q'].isin(sampled_question_ids)]
    question_with_code_description.reset_index(drop=True, inplace=True)
    l = []
    for j, code in tqdm(enumerate(question_with_code_description['Body_Q'].values)):
        if 'code description start:' in code and '<pre><code>' not in code: i += 1
        else: 
            l.append(j)
    question_with_code_description.drop(l, inplace=True)
    question_with_code_description.reset_index(drop=True, inplace=True)
    question_with_code_description.to_csv(question_with_code_description_path, index=False)






