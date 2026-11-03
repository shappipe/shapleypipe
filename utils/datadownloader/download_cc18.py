import openml
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

suite = openml.study.get_suite(99)
print(suite)

tasks = suite.tasks
print(tasks)

# save to datasets/cc18/, with files train_x.csv, train_y.csv, test_x.csv, test_y.csv
if tasks is None:
    raise ValueError('Tasks are None!')

for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    print(task)
    dataset = task.get_dataset()
    if dataset.name not in ['cmc', 'diabetes', 'jm1', 'kc1', 'kc2', 'pc4', 'phoneme', 'qsar-biodeg', 'vehicle']:
        continue
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=task.target_name # type:ignore
    )

    if str(y.dtype).lower() in ['object', 'bool']: # type:ignore
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name) # type:ignore
    
    train_indices, test_indices = task.get_train_test_split_indices()
    
    train_X = X.iloc[train_indices] # type:ignore
    train_y = y.iloc[train_indices] # type:ignore
    test_X = X.iloc[test_indices] # type:ignore
    test_y = y.iloc[test_indices] # type:ignore
    
    dataset_name = dataset.name
    dataset_dir = f"datasets/cc18/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_X.to_csv(f"{dataset_dir}/train_x.csv", index=False)
    train_y.to_csv(f"{dataset_dir}/train_y.csv", index=False)
    test_X.to_csv(f"{dataset_dir}/test_x.csv", index=False)
    test_y.to_csv(f"{dataset_dir}/test_y.csv", index=False)
    
    print(f"Dataset {dataset_name} saved to {dataset_dir}")
