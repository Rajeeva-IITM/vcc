"""Loss functions, split by the space their target lives in.

``common`` holds losses agnostic to expression-vs-velocity plus the ``CompositeLoss``
container; ``expression`` holds losses valid only on raw expression/count profiles;
``velocity`` holds losses built for flow-velocity targets. Every name is re-exported at the
top level, so existing ``from src.models.components import loss_functions as lf`` imports and
``_target_: src.models.components.loss_functions.<Class>`` config paths keep resolving.
"""

from .common import (
    AdjacencySimilarityLoss,
    BatchLaplacianReg,
    BatchVariance,
    CompositeLoss,
    GateSparsityLoss,
    GenewiseMSELoss,
    LaplacianRegularizerLoss,
    LogCoshError,
    MyMSELoss,
    WeightedContrastiveLoss,
    WeightedMAELoss,
)
from .expression import (
    MSLE,
    DiffExpAwareMSELoss,
    DiffExpError,
    DiffGeneBCELoss,
    HybridGeneLoss,
    MDPLoss,
    MSEandDiffExpLoss,
    NegativeBinomialLoss,
    SoftDiceLoss,
    SoftJaccardLoss,
)
from .velocity import (
    BatchDEAwareMSELoss,
    BatchDiffExpError,
    DEWeightedMSELoss,
)

__all__ = [
    "AdjacencySimilarityLoss",
    "BatchDEAwareMSELoss",
    "BatchDiffExpError",
    "BatchLaplacianReg",
    "BatchVariance",
    "CompositeLoss",
    "DEWeightedMSELoss",
    "DiffExpAwareMSELoss",
    "DiffExpError",
    "DiffGeneBCELoss",
    "GateSparsityLoss",
    "GenewiseMSELoss",
    "HybridGeneLoss",
    "LaplacianRegularizerLoss",
    "LogCoshError",
    "MDPLoss",
    "MSEandDiffExpLoss",
    "MSLE",
    "MyMSELoss",
    "NegativeBinomialLoss",
    "SoftDiceLoss",
    "SoftJaccardLoss",
    "WeightedContrastiveLoss",
    "WeightedMAELoss",
]
