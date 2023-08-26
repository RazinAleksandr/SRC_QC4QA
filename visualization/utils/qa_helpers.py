import pandas as pd
import plotly.offline as offline
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


classes = [
    'Data Science and Machine Learning',
    'Database and SQL',
    'GUI and Desktop Applications',
    'Networking and APIs', 
    'Other', 
    # 'Python Basics and Environment', 
    'System Administration and DevOps', 
    'Web Development',
    'API_USAGE'
    ]

def compare_metrics(dfs_adapter, dfs_pretrain):
    # Calculate mean scores for each metric and each class
    data = {
        'ROUGE_1': [df_a["ROUGE_1"].mean() / df_p["ROUGE_1"].mean() for df_a, df_p in zip(dfs_adapter, dfs_pretrain)],
        'ROUGE_2': [df_a["ROUGE_2"].mean() / df_p["ROUGE_2"].mean() for df_a, df_p in zip(dfs_adapter, dfs_pretrain)],
        'ROUGE_L': [df_a["ROUGE_L"].mean() / df_p["ROUGE_L"].mean() for df_a, df_p in zip(dfs_adapter, dfs_pretrain)],
        'BLEU': [df_a["BLEU"].mean() / df_p["BLEU"].mean() for df_a, df_p in zip(dfs_adapter, dfs_pretrain)]
    }

    df_heatmap = pd.DataFrame(data, index=classes)
    colors = ["#FF000080", "#00FF0040"]  # red with alpha=0.5, blue with alpha=0 (invisible), green with alpha=0.5
    custom_colormap = plt.matplotlib.colors.ListedColormap(colors)

    plt.figure(figsize=(16, 18))
    heatmap = sns.heatmap(df_heatmap, annot=True, cmap=custom_colormap, center=1, vmin=0, vmax=2, fmt=".4f", 
                          linewidths=.5, cbar_kws={'label': 'Ratio'}, annot_kws={"size": 20})
    
    plt.title("Relative Metrics Gain (LoRa vs base LLaMA) by Class Average Metric Scores", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20, rotation=0)  
    heatmap.figure.axes[-1].yaxis.label.set_size(20)  # Increase colorbar label font size
    
    plt.show()

def plot_cls_P_R(dfs):
    symbols = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H"]
    plt.figure(figsize=(20, 10))

    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

    for idx, df in enumerate(dfs):
        mean_rouge1 = df['ROUGE_1'].mean()
        mean_rouge2 = df['ROUGE_2'].mean()
        std_rouge1 = df['ROUGE_1'].std()
        std_rouge2 = df['ROUGE_2'].std()

        # Update bounds for zoom
        min_x = min(min_x, mean_rouge2 - std_rouge2)
        max_x = max(max_x, mean_rouge2 + std_rouge2)
        min_y = min(min_y, mean_rouge1 - std_rouge1)
        max_y = max(max_y, mean_rouge1 + std_rouge1)

        # Plotting the central point
        plt.scatter(mean_rouge2, mean_rouge1, marker=symbols[idx], s=200, label=classes[idx])

        # Plotting the standard deviation lines in red with alpha=0.5 (no arrowheads)
        plt.plot([mean_rouge2 - std_rouge2, mean_rouge2 + std_rouge2], [mean_rouge1, mean_rouge1], color='red', alpha=0.5)  # horizontal line
        plt.plot([mean_rouge2, mean_rouge2], [mean_rouge1 - std_rouge1, mean_rouge1 + std_rouge1], color='red', alpha=0.5)  # vertical line

    plt.xlabel('ROUGE 2', fontsize=20)
    plt.ylabel('ROUGE 1', fontsize=20)
    plt.title('Comparison of ROUGE scores for different classes', fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Increase tick font size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

    # Set the x and y limits for zoom
    plt.xlim(min_x - 0.02, max_x + 0.02)
    plt.ylim(min_y - 0.02, max_y + 0.02)

    plt.tight_layout()
    plt.show()

def interactive_3d_distributions(dfs):
    traces = []

    y = np.linspace(0, 1, 100)  # Assuming ROUGE_1 scores are between 0 and 1

    for idx, df in enumerate(dfs):
        z = np.histogram(df['BLEU'], bins=y, density=True)[0]
        x = [idx] * len(y)
        
        traces.append(
            go.Scatter3d(
                x=x,
                y=y[:-1],
                z=z,
                mode='lines',
                name=f'Distribution {idx}'
            )
        )

    layout = go.Layout(
        title="3D Distribution of BLEU scores for different classes",
        scene=dict(
            xaxis_title='Distribution Index',
            yaxis_title='BLEU',
            zaxis_title='Density'
        ),
        width=1200,   # Adjusting the width of the plot
        height=800    # Adjusting the height of the plot
    )
    fig = go.Figure(data=traces, layout=layout)
    
    fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.2), 
                                        eye=dict(x=1.5, y=1.5, z=0.5)))
    
    # Save the figure as an HTML file
    offline.plot(fig, filename='3d_distribution_plot.html', auto_open=False)
    fig.show()

