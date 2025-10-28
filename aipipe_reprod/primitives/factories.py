from .primitive import Primitive

from .encoders import (
    split_cat_and_num_cols, 
    NumericDataPrim,
    LabelEncoderPrim, 
    OneHotEncoderPrim)

from .fengine import (
    PolynomialFeaturesPrim, InteractionFeaturesPrim, PCA_AUTO_Prim, 
    PCA_LAPACK_Prim, PCA_ARPACK_Prim, IncrementalPCA_Prim, 
    KernelPCA_Prim, TruncatedSVD_Prim, RandomTreesEmbeddingPrim)

from .fpreprocessor import (
    MinMaxScalerPrim,
    MaxAbsScalerPrim,
    RobustScalerPrim,
    StandardScalerPrim,
    QuantileTransformerPrim,
    PowerTransformerPrim,
    LogTransformerPrim,
    NormalizerPrim,
    KBinsDiscretizerOrdinalPrim,
)

from .fselection import VarianceThresholdPrim

from .imput_cat import ImputerCatPrim

from .imput_num import ImputerMeanPrim, ImputerMedianPrim, ImputerNumPrim

from .predictors import (
    RandomForestClassifierPrim,
    AdaBoostClassifierPrim,
    BaggingClassifierPrim,
    BernoulliNBClassifierPrim,
    DecisionTreeClassifierPrim,
    ExtraTreesClassifierPrim,
    GaussianNBClassifierPrim,
    GaussianProcessClassifierPrim,
    GradientBoostingClassifierPrim,
    KNeighborsClassifierPrim,
    LinearDiscriminantAnalysisPrim,
    LinearSVCPrim,
    LogisticRegressionPrim,
    NearestCentroidPrim,
    PassiveAggressiveClassifierPrim,
    RidgeClassifierPrim,
    RidgeClassifierCVPrim,
    SGDClassifierPrim,
    SVCPrim,
    BalancedRandomForestClassifierPrim,
    EasyEnsembleClassifierPrim,
    RUSBoostClassifierPrim,
    ARDRegressionPrim,
    AdaBoostRegressorPrim,
    BaggingRegressorPrim,
    PredictorPrimitive,
)


class PrimFactory:
    prims: list[type[Primitive]] = []
    factory_name = ''

    @property
    def get_prims(self):
        return self.prims

    @property
    def get_names(self):
        return list(map(lambda x: x.get_name(), self.prims))
    
    @property
    def get_factory_name(self):
        return self.factory_name
    
    def find_prim_from_name(self, name: str):
        idx = self.get_names.index(name)
        return self.prims[idx]


class EncoderPrimFactory(PrimFactory):
    prims = [
        NumericDataPrim,
        LabelEncoderPrim, 
        OneHotEncoderPrim
    ]
    factory_name = 'Encoder'


class FenginePrimFactory(PrimFactory):
    prims = [
        PolynomialFeaturesPrim, 
        InteractionFeaturesPrim, 
        PCA_AUTO_Prim, 
        PCA_LAPACK_Prim, 
        PCA_ARPACK_Prim, 
        IncrementalPCA_Prim, 
        KernelPCA_Prim, 
        # Primitive,
        TruncatedSVD_Prim, 
        RandomTreesEmbeddingPrim,
    ]
    factory_name = 'Feature engine'

class FprocessorPrimFactory(PrimFactory):
    prims = [
        MinMaxScalerPrim,
        MaxAbsScalerPrim,
        RobustScalerPrim,
        StandardScalerPrim,
        QuantileTransformerPrim,
        # LogTransformerPrim,
        PowerTransformerPrim,
        NormalizerPrim,
        KBinsDiscretizerOrdinalPrim,
    ]
    factory_name = 'Feature processor'

class FselectionPrimFactory(PrimFactory):
    prims = [VarianceThresholdPrim]
    factory_name = 'Feature selection'

class ImputerCatPrimFactory(PrimFactory):
    prims = [ImputerCatPrim]
    factory_name = 'Imputer (category)'

class ImputerNumPrimFactory(PrimFactory):
    prims = [
        ImputerMeanPrim, 
        ImputerMedianPrim, 
        ImputerNumPrim
    ]
    factory_name = 'Imputer (numeric)'

class PredictorPrimFactory(PrimFactory):
    prims = [
        RandomForestClassifierPrim,
        AdaBoostClassifierPrim,
        BaggingClassifierPrim,
        BernoulliNBClassifierPrim,
        DecisionTreeClassifierPrim,
        ExtraTreesClassifierPrim,
        GaussianNBClassifierPrim,
        GaussianProcessClassifierPrim,
        GradientBoostingClassifierPrim,
        KNeighborsClassifierPrim,
        LinearDiscriminantAnalysisPrim,
        LinearSVCPrim,
        LogisticRegressionPrim,
        NearestCentroidPrim,
        PassiveAggressiveClassifierPrim,
        RidgeClassifierPrim,
        RidgeClassifierCVPrim,
        SGDClassifierPrim,
        SVCPrim,
        BalancedRandomForestClassifierPrim,
        EasyEnsembleClassifierPrim,
        RUSBoostClassifierPrim,
        ARDRegressionPrim,
        AdaBoostRegressorPrim,
        BaggingRegressorPrim,
    ]

    def find_prim_from_name(self, name: str):
        idx = self.get_names.index(name)
        return self.prims[idx]
