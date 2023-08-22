from tqdm import tqdm
import pandas as pd
import torch
from transformers import DataCollatorWithPadding

from data.CodeDataset import CodeDataset


# Constants
START_TAG = '<pre><code>'
END_TAG = '</code></pre>'
CODE_DESC_START = 'code description start:'
CODE_DESC_END = 'code description end'


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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer, padding=True), shuffle=False)

        for batch in tqdm(dataloader):
            with torch.no_grad():
                generated_ids = model.generate(batch["input_ids"].to(device), max_length=20)

            descriptions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, description in enumerate(descriptions):
                index = int(batch['index'][i])
                code_question_df.loc[index, 'code_description'] = str(description)

        return code_question_df

    def substitute_code_with_description(self, code_df):
        raw_df = self.df.copy()
        for row in tqdm(code_df.itertuples()):
            code_id = row.code_id
            question_id = row.question_id
            description = row.code_description
            code_snip = row.code_snippet
            
            # Find the corresponding question based on question ID
            
            try:
                question = self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'].values[0]
            except:
                print(self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'])
                print(self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'].values[0])
            
            # Replace the code with its description in the question
            question = question.replace(f'<pre><code>{code_snip}</code></pre>', f'code description start: {description} code description end')
            
            # Update the question in the question DataFrame
            self.df.loc[self.df['Id_Q'] == question_id, 'Body_Q'] = question
        
        return self.df, raw_df