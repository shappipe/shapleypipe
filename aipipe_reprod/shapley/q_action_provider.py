import numpy as np
from sklearn.metrics import accuracy_score

from ..actions.dataset_type import DatasetType
from ..primitives.predictors import LogisticRegressionPrim, MLPClassifierPrim, RandomForestClassifierPrim, XGBClassifierPrim

from ..primitives import (
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


class QActionProvider:
    '''
    Simpler version of QActionProvider.
    '''
    factories: list[PrimFactory] = [
        # ImputerCatPrimFactory(),  # 1 => index 从 0  开始
        # ImputerNumPrimFactory(),  # 3 => index 从 1  开始
        # EncoderPrimFactory(),     # 2 => index 从 4  开始
        FprocessorPrimFactory(),  # 9 => index 从 6  开始
        FenginePrimFactory(),     # 8 => index 从 15 开始
        FselectionPrimFactory(),  # 1 => index 从 24 开始
    ]
    methods: list[type[Primitive]] = []

    __cnt = 0
    spans: list[list[int]] = []
    for f in factories:
        spans.append(list(range(__cnt, __cnt + len(f.get_prims))))
        __cnt += len(f.get_prims)

    for f in factories:
        for p in f.get_prims:
            methods.append(p)

    n_action = len(methods) + 1   # 所有的算子 + 1 下游任务
    done_action = len(methods)
    action_ids = np.arange(n_action)

    # imputer_ids = (list(range(len(factories[0].get_prims))) 
    #                + list(range(len(factories[0].get_prims), len(factories[0].get_prims) + len(factories[1].get_prims))))

    # encoder_ids = list(
    #     range(len(factories[0].get_prims) + len(factories[1].get_prims), 
    #           len(factories[0].get_prims) + len(factories[1].get_prims) + len(factories[2].get_prims)))

    @classmethod
    def get(cls, action: int):
        if action < 0:
            return None
        if action == cls.done_action:
            return LogisticRegressionPrim()
        return cls.methods[action]()
    
    @classmethod
    def calculate_reward(cls, dataset: DatasetType):
        """
        计算当前的reward

        Returns:
            tuple[float, float]: 第一个是当前的reward，第二个是当前的accuracy
        """
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
