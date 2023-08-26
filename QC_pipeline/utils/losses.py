import torch.nn as nn
import torch


def cross_entropy_loss(outputs, labels, weights=None, calculate_weights=True):
    """
    Computes the Cross Entropy loss.

    Args:
        outputs (torch.Tensor): The model outputs (logits).
        labels (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The loss value.
    """
    if calculate_weights: loss_fn = nn.CrossEntropyLoss(weight=weights)
    else: loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    return loss



def binary_cross_entropy_loss(outputs, labels, weights=None, calculate_weights=False):
    """
    Computes the Binary Cross Entropy loss.

    Args:
        outputs (torch.Tensor): The model outputs (logits).
        labels (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The loss value.    
    """
    if calculate_weights: loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    else: loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(outputs, labels)
    return loss


def KLD_loss(teacher_logits, student_logits,reduction='batchmean'):
    distill_loss_fn = nn.KLDivLoss(reduction)

    student_distill_loss = distill_loss_fn(
        torch.log_softmax(student_logits, dim=-1),
        torch.softmax(teacher_logits.detach(), dim=-1)
        )
    
    # You may weigh these losses differently depending on your use case.
    #student_loss = student_task_loss + student_distill_loss
    return student_distill_loss


def focal_loss(logits, targets, gamma=0.5):
    # Reshape logits and targets for calculation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1, logits.size(-1))
    
    # Calculate sigmoid activation for logits
    probs = torch.sigmoid(logits_flat)
    
    # Calculate p for positive and 1-p for negative targets
    p = torch.where(targets_flat >= 0.5, probs, 1 - probs)
    
    # Calculate -log(p) for both positive and negative targets
    logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
    
    # Calculate the focal loss component
    loss = logp * ((1 - p) ** gamma)
    
    # Calculate the mean loss
    loss = loss.mean()
    
    return loss
