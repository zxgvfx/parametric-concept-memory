"""pcm.agent — PCM v6 Agent base layer.

Extends the F62 universal-operator architecture from passive
``(slot, Δ) → slot'`` group action to active state-action
transition ``(slot_state, action) → slot_next``. The transition
head is *literally* a F62 ``UniversalCombiner`` with an action
embedding playing the role of Δ. The policy head adds
goal-conditioned action selection on top.

See ``docs/PCM_V6_AGENT_BASE_DESIGN.md`` for the design.

Public API::

    SlotStateEncoder    — env observation → slot vector
    TransitionHead      — (slot, action) → next slot (world model)
    PolicyHead          — (slot, goal_slot) → action logits
    ValueHead           — (slot, goal_slot) → expected return
    rollout             — episode rollout helper
    bc_loss             — behavioural cloning loss
    transition_loss     — transition prediction loss

Environments::

    pcm.agent.envs.cyclic_nav.CyclicNavEnv
"""
from .heads import (
    PolicyHead,
    SlotStateEncoder,
    TransitionHead,
    ValueHead,
    bc_loss,
    transition_loss,
)
from .heads_continuous import (
    ContinuousActionRoPE,
    ContinuousPolicyHead,
    ContinuousTransitionHead,
    bc_gaussian_loss,
    continuous_transition_loss,
)
from .heads_mixed import (
    MixedActionTransitionHead,
    MixedPolicyHead,
    MultiArgActionTransitionHead,
    MultiArgPolicyHead,
    ToolEmbedding,
    mixed_bc_loss,
    mixed_transition_loss,
    multi_arg_bc_loss,
    multi_arg_transition_loss,
)
from .orchestrator import (
    continuous_rollout,
    mixed_rollout,
    rollout,
)
from .attractor_head import (
    AgentAttractorHead,
    AttractorTargets,
    agent_attractor_loss,
)
from .perception import (
    ImagePerceptionHead,
    TextPerceptionHead,
    image_to_slot,
    text_to_slot,
)
from .episodic_agent import (
    EpisodicAgent,
    EpisodicAgentDecision,
)
from .planner import (
    HybridPlanResult,
    hybrid_multi_arg_step,
    hybrid_step,
    plan_mpc,
    plan_multi_arg_mpc,
)
from .rl import (
    Episode,
    collect_episodes,
    compute_returns,
    on_policy_transition_step,
    reinforce_step,
    running_mean_baseline,
)

__all__ = [
    # v6.1 discrete
    "SlotStateEncoder",
    "TransitionHead",
    "PolicyHead",
    "ValueHead",
    "rollout",
    "bc_loss",
    "transition_loss",
    # v6.2 continuous
    "ContinuousActionRoPE",
    "ContinuousTransitionHead",
    "ContinuousPolicyHead",
    "continuous_rollout",
    "bc_gaussian_loss",
    "continuous_transition_loss",
    # v6.2-followup mixed-arity (F66)
    "ToolEmbedding",
    "MixedActionTransitionHead",
    "MixedPolicyHead",
    "mixed_rollout",
    "mixed_bc_loss",
    "mixed_transition_loss",
    # v6.2-followup² multi-arg tools (F70)
    "MultiArgActionTransitionHead",
    "MultiArgPolicyHead",
    "multi_arg_bc_loss",
    "multi_arg_transition_loss",
    # v6.3 RL closure (F67)
    "Episode",
    "collect_episodes",
    "compute_returns",
    "reinforce_step",
    "on_policy_transition_step",
    "running_mean_baseline",
    # v6.4 perception layer (F68 text + F72 image)
    "TextPerceptionHead",
    "text_to_slot",
    "ImagePerceptionHead",
    "image_to_slot",
    # v6.6 cook-then-act planner (F73)
    "AgentAttractorHead",
    "AttractorTargets",
    "agent_attractor_loss",
    "plan_mpc",
    "plan_multi_arg_mpc",
    "HybridPlanResult",
    "hybrid_step",
    "hybrid_multi_arg_step",
    # v7.2 episodic-grounded agent (F78)
    "EpisodicAgent",
    "EpisodicAgentDecision",
]
