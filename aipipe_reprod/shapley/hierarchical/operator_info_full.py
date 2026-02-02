from enum import Enum
from collections import defaultdict
import sys
import os
from dotenv import load_dotenv
# from pprint import pprint
from objprint import objprint

load_dotenv()
sys.path.append(os.getenv("PROJ_DIR"))

from aipipe_reprod.actions.q_action_provider import QActionProvider

# class OpCategory(Enum):
#     IMPUTATION = "Imputation"
#     ENCODING = "Encoding"
#     SCALING = "Scaling"
#     TRANSFORMATION = "Transformation"
#     SELECTION = "Selection"


OPERATOR_INFO: dict[int, dict[str, str]] = {
    # Imputation
    # 0: {'name': 'SimpleImputer_mean', 'category': OpCategory.IMPUTATION},
    # 1: {'name': 'SimpleImputer_median', 'category': OpCategory.IMPUTATION},
    # # Scaling
    # 2: {'name': 'StandardScaler', 'category': OpCategory.SCALING},
    # 3: {'name': 'MinMaxScaler', 'category': OpCategory.SCALING},
    # 4: {'name': 'RobustScaler', 'category': OpCategory.SCALING},
    # # Encoding
    # 5: {'name': 'OneHotEncoder', 'category': OpCategory.ENCODING},
    # # Transformation
    # 6: {'name': 'QuantileTransformer', 'category': OpCategory.TRANSFORMATION},
    # 7: {'name': 'PowerTransformer', 'category': OpCategory.TRANSFORMATION},
    # 8: {'name': 'PCA', 'category': OpCategory.TRANSFORMATION},
    # # Selection
    # 9: {'name': 'SelectKBest', 'category': OpCategory.SELECTION},
    # 10: {'name': 'RFE', 'category': OpCategory.SELECTION},
    # 18: {'name': 'SomeOtherOp', 'category': OpCategory.SELECTION},
}

for fac_id, span in enumerate(QActionProvider.spans):  # 总共6个大类
    for op_id in span:
        OPERATOR_INFO[op_id] = {'name': QActionProvider.get(op_id).get_name(), 
                                'category': QActionProvider.factories[fac_id].factory_name}
    

# 按类别组织操作
OPS_BY_CATEGORY: dict[str, list[int]] = {}
for op_idx, info in OPERATOR_INFO.items():
    OPS_BY_CATEGORY.setdefault(info['category'], []).append(op_idx)
OPS_BY_CATEGORY = dict(sorted(OPS_BY_CATEGORY.items(), key=lambda x: x[1]))
# {'Feature engine': [9, 10, 11, 12, 13, 14, 15, 16, 17],
# 'Feature processor': [0, 1, 2, 3, 4, 5, 6, 7, 8],
# 'Feature selection': [18]}

# 获取所有类别的有序列表
# CATEGORIES = sorted(list(OPS_BY_CATEGORY.keys()), key=lambda c: c)
CATEGORIES: list[str] = list(OPS_BY_CATEGORY.keys())
# ['Feature engine', 'Feature processor', 'Feature selection']

# 类别到索引的映射
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
# {'Feature engine': 0, 'Feature processor': 1, 'Feature selection': 2}

IDX_TO_CATEGORY = {i: cat for i, cat in enumerate(CATEGORIES)}
# {0: 'Feature engine', 1: 'Feature processor', 2: 'Feature selection'}

# objprint(OPERATOR_INFO)
# objprint(CATEGORIES)
# objprint(OPS_BY_CATEGORY)
# objprint(CATEGORY_TO_IDX)
# objprint(IDX_TO_CATEGORY)
