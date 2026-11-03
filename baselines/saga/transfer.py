import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from baselines.saga.apply_saga_pipeline import apply_saga_pipeline
from utils.dataloader.load_cc18 import list_cc18, load_cc18, available_cc18
from utils.dataloader.load_diffprep import name_to_df_xg, ALL_DATASET_NAMES

SEED = 0
learners = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    "XGBoost":      lambda: XGBClassifier(n_estimators=100, max_depth=6,
                                          learning_rate=0.3, random_state=SEED),
    "MLP":          lambda: MLPClassifier(hidden_layer_sizes=(100,),
                                          max_iter=100, random_state=SEED),
}

def do_cc18():
    with open('baselines/saga/pipes/cc18_pipes.json', 'r') as fin:
        pipelines = json.load(fin)

    results = []
    for ds in [*available_cc18()]:
        data = load_cc18(ds, force_factorize=True)
        X_tr, y_tr, X_te, y_te = data['train'], data['target'], data['test'], data['target_test']
        # for seed in (42, 114, 256, 512, 768):           
        pipe, X_trc, y_trc = apply_saga_pipeline(pipelines[ds]['pipelines'][0], X_tr, y_tr)
        # Xtr, Xte = pipe.transform(X_tr), pipe.transform(X_te)
        Xte = pipe.transform(X_te)
        for name, make in learners.items():
            if make is None: continue
            clf = make().fit(X_trc, y_trc)
            results.append({
                'dataset': ds,
                'learner': name,
                'acc': accuracy_score(y_te, clf.predict(Xte))
            })

    pd.DataFrame(results).to_csv('baselines/saga/transfer_results.csv', index=False, mode='a')

def do_diffprep():
    with open('baselines/saga/pipes/diffprep_pipelines.json', 'r') as fin:
        pipelines = json.load(fin)

    results = []
    for ds in ALL_DATASET_NAMES:
        data = name_to_df_xg(ds, force_factorize=True)
        X_tr, y_tr, X_te, y_te = data['train'], data['target'], data['test'], data['target_test'] 
        # for seed in (42, 114, 256, 512, 768):  
        pipe, X_trc, y_trc = apply_saga_pipeline(pipelines[ds]['pipelines'][0], X_tr, y_tr)
        # Xtr, Xte = pipe.transform(X_tr), pipe.transform(X_te)
        Xte = pipe.transform(X_te)
        for name, make in learners.items():
            if make is None: continue
            clf = make().fit(X_trc, y_trc)
            results.append({
                'dataset': ds,
                'learner': name,
                'acc': accuracy_score(y_te, clf.predict(Xte))
            })

    pd.DataFrame(results).to_csv('baselines/saga/transfer_results.csv', index=False, mode='a')

do_diffprep()
do_cc18()

'''
python baselines/saga/transfer.py
python baselines/saga/partition_results.py
'''
