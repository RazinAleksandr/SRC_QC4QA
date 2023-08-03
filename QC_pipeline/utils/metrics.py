import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



tags_classification = ['Data Science and Machine Learning', 
                       'Database and SQL', 
                       'GUI and Desktop Applications', 
                       'Networking and APIs', 
                       'Other', 
                       'Python Basics and Environment', 
                       'System Administration and DevOps',
                       'Web Development']

def accuracy(outputs, labels):
    # Convert the model outputs and labels to the appropriate format
    #preds = torch.sigmoid(outputs).data > 0.5  # Apply sigmoid and threshold at 0.5 to get binary predictions
    preds = outputs.data > 0.5
    # Calculate per-class accuracies and log them
    class_accuracies = {}
    for class_index in range(9):  # Assuming 9 classes as per your query
        class_preds = preds[:, class_index].cpu().numpy()
        class_labels = labels[:, class_index].cpu().numpy()
        accuracy = accuracy_score(class_labels, class_preds)
        # Store each class accuracy in the dictionary
        class_accuracies[f'{tags_classification[class_index]}_accuracy'] = accuracy
    
    return class_accuracies
"""
def calculate_metrics(outputs, labels):
    sigmoid_outputs = torch.sigmoid(outputs)
    preds = sigmoid_outputs.data > 0.5  
    metrics = {}
    for class_index in range(9):  # Assuming 9 classes as per your query
        class_preds = preds[:, class_index].cpu().numpy()
        class_labels = labels[:, class_index].cpu().numpy()

        precision = precision_score(class_labels, class_preds, zero_division=0)
        recall = recall_score(class_labels, class_preds, zero_division=0)
        f1 = f1_score(class_labels, class_preds, zero_division=0)

        # Store each class metrics in the dictionary
        metrics[f'{tags_classification[class_index]}_precision'] = precision
        metrics[f'{tags_classification[class_index]}_recall'] = recall
        metrics[f'{tags_classification[class_index]}_f1'] = f1

    return metrics
"""
"""def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, labels = labels.max(dim=1)

    precision = precision_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)

    # Return a dictionary containing the precision, recall, and f1 score
    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    return metrics"""

def calculate_metrics(outputs, labels):
    outputs = torch.sigmoid(outputs) >= 0.5
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = {}
    for class_index in range(8):  # Assuming 9 classes as per your query
        class_preds = outputs[:, class_index]
        class_labels = labels[:, class_index]

        precision = precision_score(class_labels, class_preds, zero_division=0)
        recall = recall_score(class_labels, class_preds, zero_division=0)
        f1 = f1_score(class_labels, class_preds, zero_division=0)

        # Store each class metrics in the dictionary
        metrics[f'{tags_classification[class_index]}_precision'] = precision
        metrics[f'{tags_classification[class_index]}_recall'] = recall
        metrics[f'{tags_classification[class_index]}_f1'] = f1

    precision = precision_score(labels, outputs, average='samples', zero_division=0)
    recall = recall_score(labels, outputs, average='samples', zero_division=0)
    f1 = f1_score(labels, outputs, average='samples', zero_division=0)

    # Return a dictionary containing the precision, recall, and f1 score
    #metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    return metrics
