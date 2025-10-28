from .factories import (
    PrimFactory, 
    ImputerMeanPrim,
    OneHotEncoderPrim,
    PolynomialFeaturesPrim,
    RobustScalerPrim,
    VarianceThresholdPrim,
)

class ImputerPrimFactory(PrimFactory):
    prims = [
        ImputerMeanPrim,
    ]
    factory_name = 'imputer'

class EncoderPrimFactory(PrimFactory):
    prims = [
        OneHotEncoderPrim
    ]
    factory_name = 'Encoder'


class FenginePrimFactory(PrimFactory):
    prims = [
        PolynomialFeaturesPrim, 
    ]
    factory_name = 'Feature engine'

class FprocessorPrimFactory(PrimFactory):
    prims = [
        RobustScalerPrim,
    ]
    factory_name = 'Feature processor'

class FselectionPrimFactory(PrimFactory):
    prims = [VarianceThresholdPrim]
    factory_name = 'Feature selection'
