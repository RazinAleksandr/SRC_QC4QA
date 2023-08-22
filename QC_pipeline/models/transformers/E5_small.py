import torch
import torch.nn as nn
import torch.nn.init as init


from transformers import AutoModel

import sys
sys.path.append('../')

# Ignore warning on model loading
from .E5_base import TextClassifier
from transformers import logging
logging.set_verbosity_error()


class SmallTextClassifier(TextClassifier):
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 8, 
                 n_inputs: int = 512,
                 batch_norm: bool = None,
                 dropout_rate: float = None,
                 activation: str = "leaky_relu",
                 **kwargs) -> None:

        # Call parent constructor
        super(SmallTextClassifier, self).__init__(model_name, 
                                                  num_classes, 
                                                  n_inputs, 
                                                  batch_norm, 
                                                  dropout_rate, 
                                                  activation, 
                                                  **kwargs)
        
        self.model = AutoModel.from_pretrained(model_name)

        # Override the classification network with a smaller one
        self.fc0 = nn.Linear(n_inputs, 128)
        # Initialize with Xavier Uniform
        init.xavier_uniform_(self.fc0.weight)
        
        if self.batch_norm: self.bn0 = nn.BatchNorm1d(128)
        if self.dropout_rate: self.dropout0 = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128, num_classes)
        # Initialize with Xavier Uniform
        init.xavier_uniform_(self.fc1.weight)

        
    def forward(self, batch_dict) -> torch.Tensor:
        # Process the input text with transformer
        outputs = self.model(**batch_dict)

        # Pool the embeddings from the transformer
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Feed the embeddings through the classification network
        out = self.fc0(embeddings)
        if self.batch_norm: out = self.bn0(out)
        out = self.activation(out)
        if self.dropout_rate: out = self.dropout0(out)

        out = self.fc1(out)

        return out
