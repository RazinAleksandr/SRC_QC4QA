from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


classes = ['Data Science and Machine Learning', 'Database and SQL', 'GUI and Desktop Applications', 
           'Networking and APIs', 'Other', 'Python Basics and Environment', 
           'System Administration and DevOps', 'Web Development']


def compute_metrics(df):
    precisions = []
    recalls = []
    f1_scores = []
    
    for i, class_name in enumerate(classes):
        true_labels = df[class_name]
        predicted_labels = df[class_name + ".1"]
        
        precisions.append(precision_score(true_labels, predicted_labels))
        recalls.append(recall_score(true_labels, predicted_labels))
        f1_scores.append(f1_score(true_labels, predicted_labels))
    
    return {
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
    }

def plot_relative_cls(df_teacher, df_student):
    teacher_metrics = compute_metrics(df_teacher)
    student_metrics = compute_metrics(df_student)

    # Calculate percentage relationship
    relative_percentage = {
        metric: [student_value / teacher_value for student_value, teacher_value in zip(student_metrics[metric], teacher_metrics[metric])]
        for metric in ['Precision', 'Recall', 'F1-Score']
    }

    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(relative_percentage, index=np.arange(len(classes)))
    df_heatmap = df_heatmap.transpose()

    # Plotting heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(df_heatmap, annot=True, cmap='RdBu_r', center=1, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Ratio'})
    plt.title('Relative Metrics Gain (Student vs Teacher) by Class')
    plt.show()

def plot_relative_cls_summarization(df1, df_summ):
    metrics = compute_metrics(df1)
    # metrics_summ = {k: [i * np.random.uniform(low=0.9, high=1.1) for i in v] for k,v in metrics.items()}
    metrics_summ = compute_metrics(df_summ)

    # Calculate percentage relationship
    relative_percentage = {
        metric: [summ_value / value for value, summ_value in zip(metrics[metric], metrics_summ[metric])]
        for metric in ['Precision', 'Recall', 'F1-Score']
    }

    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(relative_percentage, index=classes)
    
    # Define custom colormap
    colors = ["#FF000080", "#00FF0040"]  # red with alpha=0.5, blue with alpha=0 (invisible), green with alpha=0.5
    custom_colormap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Plotting heatmap
    plt.figure(figsize=(16, 18))
    heatmap = sns.heatmap(df_heatmap, annot=True, cmap=custom_colormap, center=1, vmin=0, vmax=2, fmt=".2f", 
                          linewidths=.5, cbar_kws={'label': 'Ratio'}, annot_kws={"size": 20})
    
    # Adjust font sizes
    plt.title('Relative Metrics Gain (Distilled + Summarization vs Distilled) by Class', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    heatmap.figure.axes[-1].yaxis.label.set_size(20)  # Increase colorbar label font size
    
    plt.show()

def plot_cls_metrics(df):
    metrics = compute_metrics(df)
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(metrics, index=classes)
    # Transpose the dataframe for desired heatmap layout
    df_heatmap = df_heatmap.transpose()

    # Plotting heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(df_heatmap, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Score'})
    plt.title('Metrics by Class')
    plt.show()

def plot_cls_P_R(df):
    metrics = compute_metrics(df)
    symbols = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H"]
    plt.figure(figsize=(10, 5))
    for idx, (p, r) in enumerate(zip(metrics['Precision'], metrics['Recall'])):
        plt.scatter(p, r, marker=symbols[idx], s=100, label=classes[idx])

    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Comparison of Precision/Recall scores for different classes')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()  # This ensures that everything fits well
    plt.show()


def plot_distributions(teacher_times, student_times, teacher_metrics, student_metrics, quant=0.9):
    # Compute the 0.9 quantiles for teacher and student times
    teacher_quantile = np.quantile(teacher_times, quant)
    student_quantile = np.quantile(student_times, quant)
    
    # Filter data
    teacher_times_filtered = [time for time in teacher_times if time <= teacher_quantile]
    student_times_filtered = [time for time in student_times if time <= student_quantile]

    sns.set_style("whitegrid")  # This sets a background grid which helps in reading the plot

    plt.figure(figsize=(10, 6))
    
    # KDE plot for filtered teacher times
    sns.kdeplot(teacher_times_filtered, shade=True, label=f'Teacher Times ({quant} Quantile: {teacher_quantile:.2f} seconds)', color='red')
    
    # KDE plot for filtered student times
    sns.kdeplot(student_times_filtered, shade=True, label=f'Student Times ({quant} Quantile: {student_quantile:.2f} seconds)', color='green')
    
    plt.title(f'{quant} Quantile Inference Time Distributions: Teacher (F1-score: {teacher_metrics}) vs Student (F1-score: {student_metrics})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.show()

def plot_batch_times(teacher_times, student_times, teacher_metrics, student_metrics, window_size=50):
    batches = np.arange(len(teacher_times[1:]))
    
    # Compute moving averages
    teacher_moving_avg = pd.Series(teacher_times[1:]).rolling(window=window_size, min_periods=1).mean().values
    student_moving_avg = pd.Series(student_times[1:]).rolling(window=window_size, min_periods=1).mean().values
    
    plt.figure(figsize=(12, 6))
    
    # Line plot for teacher times
    plt.plot(batches, teacher_times[1:], label=f'Teacher Times (Total: {np.sum(teacher_times):.2f}s)', color='red', alpha=0.5, linewidth=0.5)
    plt.plot(batches, teacher_moving_avg, color='red', linewidth=1.5)
    
    # Line plot for student times
    plt.plot(batches, student_times[1:], label=f'Student Times (Total: {np.sum(student_times):.2f}s)', color='green', alpha=0.5, linewidth=0.5)
    plt.plot(batches, student_moving_avg, color='green', linewidth=1.5)
    
    plt.title(f'Inference Time Per Batch: Teacher (F1-score: {teacher_metrics}) vs Student (F1-score: {student_metrics})')
    plt.xlabel('Batches')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    plt.tight_layout()  # Adjust layout for better display
    plt.show()