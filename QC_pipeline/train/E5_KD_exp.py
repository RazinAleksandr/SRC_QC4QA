import pytorch_lightning as pl
from typing import Dict, Union
from torch import Tensor
import torch
import torch.nn.functional as F
import wandb
import random

from QC_pipeline.models.transformers.E5_base import TextClassifier
from QC_pipeline.models.transformers.E5_small import SmallTextClassifier
from QC_pipeline.utils.losses import cross_entropy_loss, binary_cross_entropy_loss, KLD_loss
from QC_pipeline.utils.metrics import calculate_metrics


class E5Xperiment(pl.LightningModule):
    def __init__(self,
                 teacher_model: TextClassifier,
                 student_model: SmallTextClassifier,
                 params: Dict[str, Union[float, int]]) -> None:
        super(E5Xperiment, self).__init__()

        self.t_model = teacher_model
        self.s_model = student_model
        print(self.t_model)
        print(self.s_model)
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.freeze_st = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        if self.params['freeze']:
            # freeze transformer parameters
            for i, (name, param) in enumerate(self.t_model.named_parameters()):
                if 'model' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for i, (name, param) in enumerate(self.s_model.named_parameters()):
                if 'model' in name: #and 'model.encoder.layer.11' not in name and 'model.pooler' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        ## log params
        for name, param in self.t_model.named_parameters():
            if param.requires_grad:
                print(name)
        for name, param in self.s_model.named_parameters():
            if param.requires_grad:
                print(name)

        # Counts of each class in your training set
        # counts = [456, 1224, 578, 223, 147, 99, 587, 6, 433]
        self.classes = [j for j in range(9)]
        self.classes_name = ['Data Science and Machine Learning', 
                       'Database and SQL', 
                       'GUI and Desktop Applications', 
                       'Networking and APIs', 
                       'Other', 
                       'Python Basics and Environment', 
                       'System Administration and DevOps',
                       'Web Development']
        # weights = torch.tensor([1 / c for c in counts], dtype=torch.float32)  # inverse of the number of samples
        # self.weights = weights / weights.sum()  # normalize to sum up to 1
        # Initialize validation samples placeholder
        self.validation_samples = []
        self.epoch = 0
        self.automatic_optimization = False

    def forward_t(self, input, **kwargs) -> Tensor:
        output = self.t_model(input, **kwargs)
        #print(output)
        return output 
    
    def forward_s(self, input, **kwargs) -> Tensor:
        output = self.s_model(input, **kwargs)
        #print(output)
        return output

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch_dict, y_batch, _ = batch
        self.curr_device = y_batch.device

        t_opt, s_opt = self.optimizers()
        t_scheduler, s_scheduler = self.lr_schedulers() #### !!!!!!

        # Teacher
        teacher_output = self.forward_t(batch_dict)
        teacher_loss = binary_cross_entropy_loss(teacher_output, y_batch)
        t_opt.zero_grad()
        self.manual_backward(teacher_loss)
        # clip gradients
        self.clip_gradients(t_opt, gradient_clip_val=1, gradient_clip_algorithm="value")
        t_opt.step()

        # Student
        student_output = self.forward_s(batch_dict)
        student_loss = binary_cross_entropy_loss(student_output, y_batch)
        student_distil_loss = KLD_loss(
            F.log_softmax(teacher_output / self.params["temperature"], dim=-1),
            F.log_softmax(student_output / self.params["temperature"], dim=-1),
            )
        total_student_loss = (1 - self.params["distill_weight"]) * student_loss + self.params["distill_weight"] * student_distil_loss * (self.params["temperature"] ** 2)
        s_opt.zero_grad()
        self.manual_backward(total_student_loss)
        # clip gradients
        self.clip_gradients(s_opt, gradient_clip_val=1, gradient_clip_algorithm="value")
        s_opt.step()

        if self.current_epoch - self.epoch >= 1:
            t_scheduler.step()
            s_scheduler.step()  # update the student's scheduler
            self.epoch += 1
        
        # if self.current_epoch == 3 and self.freeze_st == False:
        #     for i, (name, param) in enumerate(self.s_model.named_parameters()):
        #         param.requires_grad = True
        #     self.freeze_st = True


        # Log
        self.log_dict({
            "t_ce_loss": teacher_loss, 
            "s_ce_loss": student_loss,
            "s_kld_loss": student_distil_loss,
            "s_total_loss": total_student_loss,
            
            }, 
            #prog_bar=True, 
            on_step=True)

    def validation_step(self, batch: Tensor, batch_idx: int, optimizer_idx: int = 0):
        batch_dict, y_batch, text = batch
        self.curr_device = y_batch.device

        # Teacher
        teacher_output = self.forward_t(batch_dict)
        t_ce_val_loss = binary_cross_entropy_loss(teacher_output, y_batch)
        t_vall_metrics = calculate_metrics(teacher_output, y_batch)

        # Student
        student_output = self.forward_s(batch_dict)
        s_ce_val_loss = binary_cross_entropy_loss(student_output, y_batch)
        s_distil_val_loss = KLD_loss(
            F.log_softmax(teacher_output / self.params["temperature"], dim=-1),
            F.log_softmax(student_output / self.params["temperature"], dim=-1),
            )
        s_total_val_loss = (1 - self.params["distill_weight"]) * s_ce_val_loss + self.params["distill_weight"] * s_distil_val_loss * (self.params["temperature"] ** 2)
        s_vall_metrics = calculate_metrics(student_output, y_batch)
        

        #############################
        # Get predictions
        t_preds = torch.sigmoid(teacher_output) >= 0.5
        s_preds = torch.sigmoid(student_output) >= 0.5

        # choose random row from this batch
        random_index = random.choice(range(teacher_output.shape[0]))
        true_classes = y_batch[random_index]

        t_predicted_classes = t_preds[random_index]
        s_predicted_classes = s_preds[random_index]
        

        # Convert binary outputs to class names
        t_predicted_class_names = [self.classes_name[i] for i, x in enumerate(t_predicted_classes) if x]
        s_predicted_class_names = [self.classes_name[i] for i, x in enumerate(s_predicted_classes) if x]

        true_class_names = [self.classes_name[i] for i, x in enumerate(true_classes) if x]

        # For text data use text[random_index]
        self.validation_samples.append([random_index, text[random_index], t_predicted_class_names, s_predicted_class_names, true_class_names])
        # self.validation_samples.append([random_index, text[random_index], t_predicted_class_names, true_class_names])
        #############
        
        # Log
        self.log_dict({
            "t_ce_val_loss": t_ce_val_loss, 
            "s_ce_val_loss": s_ce_val_loss,
            "s_kld_val_loss": s_distil_val_loss,
            "s_total_val_loss": s_total_val_loss,
            }, 
            on_epoch=True)
        
        t_vall_metrics_log = {f'{k}_t': v for k, v in t_vall_metrics.items()}
        s_vall_metrics_log = {f'{k}_s': v for k, v in s_vall_metrics.items()}

        self.log_dict(t_vall_metrics_log, on_epoch=True)
        self.log_dict(s_vall_metrics_log, on_epoch=True)

    def on_validation_epoch_end(self):
        # Log the validation samples as a table
        self.logger.experiment.log({"validation_samples": wandb.Table(data=self.validation_samples, columns=["id", "Text", "T_Prediction", "S_Prediction", "Ground Truth"])})
        # self.logger.experiment.log({"validation_samples": wandb.Table(data=self.validation_samples, columns=["id", "Text", "T_Prediction", "Ground Truth"])})
        
        # Clear the validation samples for the next epoch
        self.validation_samples = []

    def configure_optimizers(self):
        t_opt = torch.optim.Adam(
            self.t_model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        s_opt = torch.optim.Adam(
            self.s_model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        
        # Assuming that you want to use the same learning rate scheduler for both optimizers
        t_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(t_opt, gamma=self.params['scheduler_gamma']),
            'monitor': 'val_loss',  # Assuming you want to schedule based on validation loss
        }

        s_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(s_opt, gamma=self.params['scheduler_gamma']),
            'monitor': 'val_loss',  # Assuming you want to schedule based on validation loss
        }

        return [t_opt, s_opt], [t_scheduler, s_scheduler]
