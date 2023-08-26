import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel, logging
logging.set_verbosity_error()

from QC_pipeline.models.transformers.E5_base import TextClassifier


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
        
        if self.batch_norm: 
            self.bn0 = nn.BatchNorm1d(256)
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(32)
        if self.dropout_rate: 
            self.dropout0 = nn.Dropout(dropout_rate)

        # Define the layers for the classification network
        self.fc0 = nn.Linear(n_inputs, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        del self.fc5
        # Initialize with Xavier Uniform
        init.xavier_uniform_(self.fc0.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)

        
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
        if self.batch_norm: out = self.bn1(out)
        out = self.activation(out)
        if self.dropout_rate: out = self.dropout0(out)

        out = self.fc2(out)
        if self.batch_norm: out = self.bn2(out)
        out = self.activation(out)
        if self.dropout_rate: out = self.dropout0(out)

        out = self.fc3(out)
        if self.batch_norm: out = self.bn3(out)
        out = self.activation(out)
        if self.dropout_rate: out = self.dropout0(out)

        out = self.fc4(out)

        return out
