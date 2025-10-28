from enum import Enum
from collections import defaultdict
import sys
import os
from dotenv import load_dotenv
# from pprint import pprint
from objprint import objprint

load_dotenv()
sys.path.append(os.getenv("PROJ_DIR"))

from aipipe_reprod.new_ql.q_action_provider import QActionProvider


OPERATOR_INFO: dict[int, dict[str, str]] = {
}

for fac_id, span in enumerate(QActionProvider.spans):
    for op_id in span:
        OPERATOR_INFO[op_id] = {'name': QActionProvider.get(op_id).get_name(), 
                                'category': QActionProvider.factories[fac_id].factory_name}
    

OPS_BY_CATEGORY: dict[str, list[int]] = {}
for op_idx, info in OPERATOR_INFO.items():
    OPS_BY_CATEGORY.setdefault(info['category'], []).append(op_idx)
OPS_BY_CATEGORY = dict(sorted(OPS_BY_CATEGORY.items(), key=lambda x: x[1]))

CATEGORIES: list[str] = list(OPS_BY_CATEGORY.keys())


CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}


IDX_TO_CATEGORY = {i: cat for i, cat in enumerate(CATEGORIES)}


# objprint(OPERATOR_INFO)
# objprint(CATEGORIES)
# objprint(OPS_BY_CATEGORY)
# objprint(CATEGORY_TO_IDX)
# objprint(IDX_TO_CATEGORY)
