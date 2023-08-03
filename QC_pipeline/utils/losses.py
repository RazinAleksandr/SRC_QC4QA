import torch.nn as nn


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


