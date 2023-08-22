from torch.utils.data import Dataset
import torch


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