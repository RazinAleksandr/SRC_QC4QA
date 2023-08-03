import argparse
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.utils import shuffle
from os.path import join as opj
import numpy as np
from . import DataPreprocessing
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing and splitting StackOverflow questions dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)

    # Load raw data
    text_qa = pd.read_csv(config['raw_data_path'])
    print(f"Raw data shape: {text_qa.shape}")

    # Perform preprocessing
    preprocessor = DataPreprocessing(text_qa)
    preprocessor.perform_preprocessing()
    preprocessed_df = preprocessor.preprocess_QC4QA()
    print(f"Processed data shape: {preprocessed_df.shape}")

    # Split the data
    #train_df, val_df = train_test_split(preprocessed_df, test_size=config['val_size'], shuffle=config['shuffle'], random_state=42)
    # Convert your DataFrame column to a numpy array
    X = preprocessed_df['Text'].values
    y = np.stack(preprocessed_df['Label'].values)

    # Shuffle your data
    X, y = shuffle(X, y, random_state=42)
    # Split data
    X_train, y_train, X_test, y_test = iterative_train_test_split(X.reshape(-1, 1), y, test_size=config['val_size'])
    # Data to dfs
    train_df = pd.DataFrame({'Text': X_train.flatten(), 'Label': list(y_train)})
    val_df = pd.DataFrame({'Text': X_test.flatten(), 'Label': list(y_test)})

    # print train set info
    class_counts = train_df['Label'].values.sum(axis=0)
    print('Train set')
    for i, count in enumerate(class_counts):
        print(f"Class {i} count: {count}")

    # print val set info
    class_counts = val_df['Label'].values.sum(axis=0)
    print('Val set')
    for i, count in enumerate(class_counts):
        print(f"Class {i} count: {count}")

    # Save to CSV
    train_df.to_csv(opj(config['processed_data_path'], 'train.csv'), index=False)
    val_df.to_csv(opj(config['processed_data_path'], 'val.csv'), index=False)

if __name__ == "__main__":
    main()
