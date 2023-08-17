from tqdm import tqdm
import torch
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

from data.CodeDataset import CodeDataset
from utils.helpers import extract_code_snippets_from_text


# Constants
START_TAG = '<pre><code>'
END_TAG = '</code></pre>'
CODE_DESC_START = 'code description start:'
CODE_DESC_END = 'code description end'


class DataPreprocessing:
    def __init__(self, dataframe, columns=None):
        if columns is None:
            columns = ['Id_Q', 'Title_Q', 'Body_Q', 'Tags_Q', 'Code_Q']
        self.df = dataframe[columns].drop_duplicates()

    def preprocess_tags(self):
        """Split comma-separated tags into lists."""
        self.df['Tags_Q'] = self.df['Tags_Q'].str.split(',')

    def extract_code_questions(self):
        """Extract code snippets from questions."""
        self.df['code_snippets'] = self.df['Body_Q'].apply(extract_code_snippets_from_text)
        self.code_df = self.df[self.df['code_snippets'].str.len() > 0]


    def generate_code_descriptions(self, tokenizer, model, batch_size=64):
        """Generate descriptions for code snippets."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dataset = CodeDataset(self.code_df, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer, padding=True))

        descriptions = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                generated_ids = model.generate(batch["input_ids"].to(device), max_length=20)
                descriptions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        
        return descriptions

    def substitute_code_with_description(self, descriptions):
        """Replace code snippets with their generated descriptions."""
        for idx, code_list in enumerate(self.code_df['code_snippets']):
            for code, desc in zip(code_list, descriptions):
                self.df['Body_Q'] = self.df['Body_Q'].str.replace(f'{START_TAG}{code}{END_TAG}', f'{CODE_DESC_START} {desc} {CODE_DESC_END}')
        return self.df