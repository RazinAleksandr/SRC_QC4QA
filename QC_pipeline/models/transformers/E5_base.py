import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor
import torch.nn.functional as F

from transformers import AutoModel
from transformers import BertModel

# Ignore warning on model loading
from transformers import logging
logging.set_verbosity_error()


class TextClassifier(nn.Module):
    """
    Defines the architecture of the text classifier, which includes a pre-trained Transformer model for embeddings,
    and a classification network on top.
    """

    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 8, 
                 n_inputs: int = 512,
                 batch_norm: bool = None,
                 dropout_rate: float = None,
                 activation: str = "leaky_relu",
                 **kwargs) -> None:
        """
        Initializes the TextClassifier object.

        Args:
            model_name (str): The name of the transformer model.
            num_classes (int, optional): Number of classes for the output layer. Defaults to 8.
            n_inputs (int, optional): Number of input features for the FC layer. Defaults to 512.
        """
        super(TextClassifier, self).__init__()

        # Load the transformer model
        self.model = AutoModel.from_pretrained(model_name)
        # Load the BERT model
        #self.model = BertModel.from_pretrained(model_name)

        self.batch_norm = batch_norm  # Flag to indicate if batch normalization is applied
        self.dropout_rate = dropout_rate  # Probability of dropout
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
        }
        self.activation = activations.get(activation, None)  # Activation function for the model

        if self.batch_norm: 
            self.bn0 = nn.BatchNorm1d(512)
            self.bn1 = nn.BatchNorm1d(256)
            # self.bn2 = nn.BatchNorm1d(128)
            # self.bn3 = nn.BatchNorm1d(64)
            # self.bn4 = nn.BatchNorm1d(32)
        if self.dropout_rate: 
            self.dropout0 = nn.Dropout(dropout_rate)
        

        # Define the layers for the classification network
        self.fc0 = nn.Linear(n_inputs, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 32)
        # self.fc5 = nn.Linear(32, num_classes)
        
        # Initialize with Xavier Uniform
        init.xavier_uniform_(self.fc0.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        # init.xavier_uniform_(self.fc3.weight)
        # init.xavier_uniform_(self.fc4.weight)
        # init.xavier_uniform_(self.fc5.weight)
        # Creating multiple output heads, one for each class
        #self.fcs = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_classes)])

    def average_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        """
        Applies an average pooling operation over the hidden states, accounting for attention masks.

        Args:
            last_hidden_states (Tensor): Last hidden states from the transformer model.
            attention_mask (Tensor): Attention masks for the input sequences.

        Returns:
            Tensor: Pooled embeddings.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, batch_dict) -> torch.Tensor:
        """
        Defines the forward pass for the classification network.

        Args:
            batch_dict (dict): Dictionary containing input tensors.

        Returns:
            torch.Tensor: Output from the network.
        """
        # Process the input text with transformer
        outputs = self.model(**batch_dict)

        # Pool the embeddings from the transformer
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        #embeddings = self.model(**batch_dict)[0][:, 0, :]

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
        # if self.batch_norm: out = self.bn2(out)
        # out = self.activation(out)
        # if self.dropout_rate: out = self.dropout0(out)

        # out = self.fc3(out)
        # if self.batch_norm: out = self.bn3(out)
        # out = self.activation(out)
        # if self.dropout_rate: out = self.dropout0(out)

        # out = self.fc4(out)
        # if self.batch_norm: out = self.bn4(out)
        # out = self.activation(out)
        # if self.dropout_rate: out = self.dropout0(out)

        # out = self.fc5(out)
        return out
        #out = [F.sigmoid(fc(out)) for fc in self.fcs]
        #out = [fc(out) for fc in self.fcs]  # Removed the sigmoid function

        #return torch.cat(out, dim=1)  # Concatenate the outputs

# class TextClassifier(nn.Module):
#     """
#     Defines the architecture of the text classifier, which includes a pre-trained Transformer model for embeddings,
#     and a classification network on top.
#     """

#     def __init__(self, 
#                  model_name: str, 
#                  num_classes: int = 8, 
#                  n_inputs: int = 512,
#                  batch_norm: bool = None,
#                  dropout_rate: float = None,
#                  activation: str = "leaky_relu",
#                  **kwargs) -> None:
#         """
#         Initializes the TextClassifier object.

#         Args:
#             model_name (str): The name of the transformer model.
#             num_classes (int, optional): Number of classes for the output layer. Defaults to 8.
#             n_inputs (int, optional): Number of input features for the FC layer. Defaults to 512.
#         """
#         super(TextClassifier, self).__init__()

#         # Load the transformer model
#         self.model = AutoModel.from_pretrained(model_name)
#         # Load the BERT model
#         #self.model = BertModel.from_pretrained(model_name)

#         self.batch_norm = batch_norm  # Flag to indicate if batch normalization is applied
#         self.dropout_rate = dropout_rate  # Probability of dropout
#         activations = {
#             "relu": nn.ReLU(),
#             "leaky_relu": nn.LeakyReLU(),
#             "elu": nn.ELU(),
#             "gelu": nn.GELU(),
#             "selu": nn.SELU(),
#         }
#         self.activation = activations.get(activation, None)  # Activation function for the model


#         # Define the layers for the classification network
#         self.fc0 = nn.Linear(n_inputs, 512)
#         # Initialize with Xavier Uniform
#         init.xavier_uniform_(self.fc0.weight)

#         if self.batch_norm: self.bn0 = nn.BatchNorm1d(512)
#         if self.dropout_rate: self.dropout0 = nn.Dropout(dropout_rate)
        
#         self.fc1 = nn.Linear(512, 256)
#         # Initialize with Xavier Uniform
#         init.xavier_uniform_(self.fc1.weight)
        
#         if self.batch_norm: self.bn1 = nn.BatchNorm1d(256)
#         if self.dropout_rate: self.dropout1 = nn.Dropout(dropout_rate)
        
#         self.fc2 = nn.Linear(256, num_classes)
#         # Initialize with Xavier Uniform
#         init.xavier_uniform_(self.fc2.weight)
#         # Creating multiple output heads, one for each class
#         #self.fcs = nn.ModuleList([nn.Linear(256, 1) for _ in range(num_classes)])

#     def average_pool(self, last_hidden_states: Tensor,
#                      attention_mask: Tensor) -> Tensor:
#         """
#         Applies an average pooling operation over the hidden states, accounting for attention masks.

#         Args:
#             last_hidden_states (Tensor): Last hidden states from the transformer model.
#             attention_mask (Tensor): Attention masks for the input sequences.

#         Returns:
#             Tensor: Pooled embeddings.
#         """
#         last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

#     def forward(self, batch_dict) -> torch.Tensor:
#         """
#         Defines the forward pass for the classification network.

#         Args:
#             batch_dict (dict): Dictionary containing input tensors.

#         Returns:
#             torch.Tensor: Output from the network.
#         """
#         # Process the input text with transformer
#         outputs = self.model(**batch_dict)

#         # Pool the embeddings from the transformer
#         embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#         #embeddings = self.model(**batch_dict)[0][:, 0, :]

#         # Feed the embeddings through the classification network
#         out = self.fc0(embeddings)
#         if self.batch_norm: out = self.bn0(out)
#         out = self.activation(out)
#         if self.dropout_rate: out = self.dropout0(out)

#         out = self.fc1(out)
#         if self.batch_norm: out = self.bn1(out)
#         out = self.activation(out)
#         if self.dropout_rate: out = self.dropout1(out)

#         out = self.fc2(out)
#         return out
#         #out = [F.sigmoid(fc(out)) for fc in self.fcs]
#         #out = [fc(out) for fc in self.fcs]  # Removed the sigmoid function

#         #return torch.cat(out, dim=1)  # Concatenate the outputs