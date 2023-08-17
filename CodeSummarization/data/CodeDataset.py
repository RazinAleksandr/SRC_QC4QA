from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.code_snippets = dataframe['code_snippets'].explode().dropna().tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, idx):
        code_snippet = self.code_snippets[idx]
        encoded_input = self.tokenizer.encode_plus(code_snippet, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {"input_ids": encoded_input["input_ids"].squeeze()}

