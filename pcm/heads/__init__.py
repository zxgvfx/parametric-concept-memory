"""Muscle heads - task-specific modules that consume ParamBundle facets.

The module exports two parallel sets of public symbols:

- **Tier-A heads** (``ArithmeticHeadV2``, ``ComparisonHead``,
  ``NumerosityClassifier``, ...) - the project's ``nn.Module`` muscles.
  Their ``fc1``/``fc2``/``fc3`` Linear layers are the storage of the
  trainable backbone weights and are reused by the Tier-D cook path
  via :class:`HeadAsBackbone` (parameter sharing - no extra parameters).
- **Tier-D cook foundations** (:mod:`pcm.heads.cook_factory`) and
  **per-head cook wrappers** (:mod:`pcm.heads.cook_wrappers`) -
  ``build_*_cook(cg, head)`` functions that wrap a Tier-A head into a
  ``parametric_muscle_subgraph`` ConceptNode that is cooked through
  :class:`pcm.GraphEvaluator`. Every wrapper is bit-identical to the
  direct-forward path (claim D1, see ``tests/test_cook_all_heads.py``).
"""
from .arithmetic_head_v2 import ArithmeticHeadV2
from .comparison_head import ComparisonHead
from .cook_factory import (
    HeadAsBackbone,
    MLPBackbone,
    copy_three_linears,
    make_cook_subgraph,
)
from .cook_wrappers import (
    build_arith_v2_cook,
    build_comparison_cook,
    build_numerosity_classifier_cook,
)
from .numerosity_classifier import NumerosityClassifier
from .numerosity_encoder import (
    DatasetConfig,
    NumerosityEncoder,
    encode_numerosity,
    generate_dot_canvas,
)

__all__ = [
    "ArithmeticHeadV2",
    "ComparisonHead",
    "NumerosityClassifier",
    "NumerosityEncoder", "DatasetConfig",
    "encode_numerosity", "generate_dot_canvas",
    "MLPBackbone", "HeadAsBackbone",
    "copy_three_linears", "make_cook_subgraph",
    "build_arith_v2_cook",
    "build_comparison_cook",
    "build_numerosity_classifier_cook",
]
