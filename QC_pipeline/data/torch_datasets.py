from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
import ast


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset class for the text classification task.
    """

    def __init__(self, data_path: str, tokenizer_name: str, max_len: int = 512):
        """
        Initializes the TextClassificationDataset object.

        Args:
            dataframe (str): Path to the dataframe containing the text data and labels.
            tokenizer (str): Name of tokenizer corresponding to the transformer model.
            max_len (int, optional): Maximum length for the tokenized sequences. Defaults to 512.
        """
        self.dataframe = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        """
        Returns the length of the dataframe.

        Returns:
            int: Length of the dataframe.
        """
        return len(self.dataframe)

    def __getitem__(self, index: int):
        """
        Returns a dictionary with the tokenized sequence and corresponding label for the given index.

        Args:
            index (int): The index for which to return the data.

        Returns:
            dict: Contains the tokenized sequence and the corresponding label.
        """
        # Select the text and label at the specified index
        text = self.dataframe.iloc[index]['Text']
        text = 'query: ' + text
        label = self.dataframe.iloc[index]['Label']
        label = label.replace(" ", ", ")
        label = ast.literal_eval(label)

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Return a dictionary containing the tokenized text and the label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }, torch.tensor(label, dtype=torch.float), text

