from ..factories import PrimFactory

from .outlier import (
    IsolationForestFilterPrim,
    LOFFilterPrim,
    WinsorizerPrim,
    ZScoreClipperPrim,
)
from .imbalance import (
    SMOTEPrim,
    RandomUnderSamplerPrim,
    RandomOverSamplerPrim,
)


class OutlierPrimFactory(PrimFactory):
    prims = [
        IsolationForestFilterPrim,   # 25
        LOFFilterPrim,               # 26
        WinsorizerPrim,              # 27
        ZScoreClipperPrim,           # 28
    ]
    factory_name = 'Outlier handling'


class ImbalancePrimFactory(PrimFactory):
    prims = [
        SMOTEPrim,                   # 29
        RandomUnderSamplerPrim,      # 30
        RandomOverSamplerPrim,       # 31
    ]
    factory_name = 'Imbalance handling'


EXT_FACTORIES = [OutlierPrimFactory(), ImbalancePrimFactory()]

EXT_BASE_ID = 25
OUTLIER_IDS = [25, 26, 27, 28]
IMBALANCE_IDS = [29, 30, 31]

ROW_CHANGING_IDS = [25, 26, 29, 30, 31]
