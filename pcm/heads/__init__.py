"""Muscle heads — task-specific modules that consume ParamBundle facets."""
from .arithmetic_head import ArithmeticHead
from .arithmetic_head_v2 import ArithmeticHeadV2
from .comparison_head import ComparisonHead
from .numerosity_classifier import NumerosityClassifier
from .numerosity_encoder import (
    DatasetConfig,
    NumerosityEncoder,
    encode_numerosity,
    generate_dot_canvas,
)

__all__ = [
    "ArithmeticHead",
    "ArithmeticHeadV2",
    "ComparisonHead",
    "NumerosityClassifier",
    "NumerosityEncoder",
    "DatasetConfig",
    "encode_numerosity",
    "generate_dot_canvas",
]
