import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Union, List, Tuple
from torch import Tensor
import torch
import wandb
import random

from . import TextClassifier, cross_entropy_loss, calculate_metrics, binary_cross_entropy_loss


class E5Xperiment(pl.LightningModule):
    def __init__(self,
                 model: TextClassifier,
                 params: Dict[str, Union[float, int]]) -> None:
        super(E5Xperiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        if self.params['freeze']:
            # freeze transformer parameters
            for i, (name, param) in enumerate(model.named_parameters()):
                if 'model' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # Counts of each class in your training set
        counts = [456, 1224, 578, 223, 147, 99, 587, 6, 433]
        self.classes = [j for j in range(9)]
        self.classes_name = ['Data Science and Machine Learning', 
                       'Database and SQL', 
                       'GUI and Desktop Applications', 
                       'Networking and APIs', 
                       'Other', 
                       'Python Basics and Environment', 
                       'System Administration and DevOps',
                       'Web Development']
        weights = torch.tensor([1 / c for c in counts], dtype=torch.float32)  # inverse of the number of samples
        self.weights = weights / weights.sum()  # normalize to sum up to 1
        # Initialize validation samples placeholder
        self.validation_samples = []

    def forward(self, input, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch: Tensor, batch_idx: int, optimizer_idx: int = 0) -> Tensor:
        batch_dict, y_batch, _ = batch
        
        self.curr_device = y_batch.device

        results = self.forward(batch_dict)
        # Send the weights to the same device as your model
        weights = self.weights.to(self.curr_device)

        train_loss = binary_cross_entropy_loss(
            results, y_batch)
            #results, y_batch, weights=weights, calculate_weights=True)
        
        self.log("train_loss", train_loss, on_step=True)
        return train_loss

    def validation_step(self, batch: Tensor, batch_idx: int, optimizer_idx: int = 0):
        batch_dict, y_batch, text = batch
        
        self.curr_device = y_batch.device

        results = self.forward(batch_dict)
        val_loss = binary_cross_entropy_loss(
            results, y_batch)
        vall_metrics = calculate_metrics(results, y_batch)
        #############################
        """# Get predictions
        _, preds = torch.max(results, dim=1)
        _, labels = y_batch.max(dim=1)
        # choose random row from this batch
        random_index = random.choice(range(preds.shape[0]))
        predicted_class = preds[random_index].item()
        true_class = labels[random_index].item()

        # For text data use text[random_index]
        self.validation_samples.append([random_index, text[random_index], predicted_class, true_class])"""
        # Get predictions
        preds = torch.sigmoid(results) >= 0.5

        # choose random row from this batch
        random_index = random.choice(range(preds.shape[0]))
        predicted_classes = preds[random_index]
        true_classes = y_batch[random_index]

        # Convert binary outputs to class names
        predicted_class_names = [self.classes_name[i] for i, x in enumerate(predicted_classes) if x]
        true_class_names = [self.classes_name[i] for i, x in enumerate(true_classes) if x]

        # For text data use text[random_index]
        self.validation_samples.append([random_index, text[random_index], predicted_class_names, true_class_names])
        #############

        self.log_dict(vall_metrics, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def on_validation_epoch_end(self):
        # Log the validation samples as a table
        self.logger.experiment.log({"validation_samples": wandb.Table(data=self.validation_samples, columns=["id", "Text", "Prediction", "Ground Truth"])})
        # Clear the validation samples for the next epoch
        self.validation_samples = []

    
    def configure_optimizers(self) -> Union[List[optim.Optimizer], Tuple[List[optim.Optimizer], List[_LRScheduler]]]:

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims