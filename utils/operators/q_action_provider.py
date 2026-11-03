import numpy as np
from sklearn.metrics import accuracy_score

from ..datatypes import DatasetType
from .primitives.predictors import LogisticRegressionPrim, RandomForestClassifierPrim, MLPClassifierPrim, GradientBoostingClassifierPrim, XGBClassifierPrim

from .primitives import (
    PrimFactory,
    ImputerCatPrimFactory,
    ImputerNumPrimFactory,
    EncoderPrimFactory,
    FprocessorPrimFactory,
    FenginePrimFactory,
    FselectionPrimFactory,
    PredictorPrimFactory,

    Primitive,
)

import os as _os

EXT_OPS_ENABLED = _os.getenv('SHAPLEY_EXT_OPS', '0') == '1'

_BASE_FACTORIES: list = [
    ImputerCatPrimFactory(), 
    ImputerNumPrimFactory(), 
    EncoderPrimFactory(),    
    FprocessorPrimFactory(), 
    FenginePrimFactory(),    
    FselectionPrimFactory(), 
]

_EXT_EXCLUSIVE: list[list[int]] = []
if EXT_OPS_ENABLED:
    from .primitives.ext.factories_ext import EXT_FACTORIES
    _BASE_FACTORIES = _BASE_FACTORIES + list(EXT_FACTORIES)
    _EXT_EXCLUSIVE = [[25, 26], [29, 30, 31]]


class QActionProvider:
    factories: list[PrimFactory] = _BASE_FACTORIES
    methods: list[type[Primitive]] = []

    __cnt = 0
    spans: list[list[int]] = []
    for f in factories:
        spans.append(list(range(__cnt, __cnt + len(f.get_prims))))
        __cnt += len(f.get_prims)

    for f in factories:
        for p in f.get_prims:
            methods.append(p)

    exclusive_methods = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
        [15, 16, 23],
        [17, 18, 19, 20, 21, 22],
    ] + _EXT_EXCLUSIVE

    n_action = len(methods) + 1
    done_action = len(methods)
    action_ids = np.arange(n_action)

    imputer_ids = (list(range(len(factories[0].get_prims))) 
                   + list(range(len(factories[0].get_prims), len(factories[0].get_prims) + len(factories[1].get_prims))))

    encoder_ids = list(
        range(len(factories[0].get_prims) + len(factories[1].get_prims), 
              len(factories[0].get_prims) + len(factories[1].get_prims) + len(factories[2].get_prims)))

    @classmethod
    def get(cls, action: int):
        if action < 0:
            raise ValueError(f'Cannot get an action id less than 0! {action=}')
        if action == cls.done_action:
            return LogisticRegressionPrim()
        return cls.methods[action]()

    @classmethod
    def remove_invalid_actions(cls, selected_action: int, valid_action_ids: list[int]):
        if selected_action == cls.done_action:
            return valid_action_ids
        for exclusive_group in cls.exclusive_methods:
            if selected_action in exclusive_group:
                valid_action_ids = list(set(valid_action_ids) - set(exclusive_group))
        return valid_action_ids
    
    @classmethod
    def calculate_reward(cls, dataset: DatasetType):
        if dataset is None:
            return 0., 0.
        
        if dataset['train'].empty or dataset['test'].empty:
            return 0., 0.
        
        if dataset['train'].shape[1] > 100:
            d = {'train': dataset['train'].iloc[:, :100],
                 'test': dataset['test'].iloc[:, :100],
                 'target': dataset['target'],
                 'target_test': dataset['target_test']}
        else:
            d = dataset

        predictor = LogisticRegressionPrim()
        y_pred = predictor.transform_1(d['train'], d['target'], d['test'])
        accuracy = accuracy_score(d['target_test'], y_pred)
        return accuracy, accuracy
        
    @classmethod
    def str_to_idx(cls, s: str):
        s = s.replace('<', '').replace('>', '')
        if s == 'blank':
            return QActionProvider.done_action
        for i, m in enumerate(cls.methods):
            if m().__class__.__name__.__contains__(s) or s.__contains__(m.__name__):
                return i
        return QActionProvider.done_action
    
    @classmethod
    def idx_to_factory_id(cls, idx: int):
        for i, span in enumerate(cls.spans):
            if idx in span:
                return i
        return -1
    
    @classmethod
    def rf(cls):
        return RandomForestClassifierPrim()
    
    @classmethod
    def lr(cls):
        return LogisticRegressionPrim()
    
    @classmethod
    def mlp(cls):
        return MLPClassifierPrim()

    @classmethod
    def xg(cls):
        return XGBClassifierPrim()
