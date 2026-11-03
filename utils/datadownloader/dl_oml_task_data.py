import json
import sys
import os
import openml
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

def download_task_dataset(task_id: int, output_dir: str):  # covertype task_id=2910
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    print(dataset)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=task.target_name #type: ignore
    )

    train_indices, test_indices = task.get_train_test_split_indices()
    
    train_X = X.iloc[train_indices] #type: ignore
    train_y = y.iloc[train_indices] #type: ignore
    test_X = X.iloc[test_indices] #type: ignore
    test_y = y.iloc[test_indices] #type: ignore

    dataset_name = dataset.name
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    train_X.to_csv(f"{dataset_dir}/train_x.csv", index=False)
    train_y.to_csv(f"{dataset_dir}/train_y.csv", index=False)
    test_X.to_csv(f"{dataset_dir}/test_x.csv", index=False)
    test_y.to_csv(f"{dataset_dir}/test_y.csv", index=False)
    
    print(f"Dataset {dataset_name} saved to {dataset_dir}")

import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task_id', type=int, required=True)
    argparser.add_argument('--output_dir', type=str, required=True)
    args = argparser.parse_args()
    download_task_dataset(args.task_id, args.output_dir)
