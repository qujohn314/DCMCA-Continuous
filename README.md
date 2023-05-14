# DCMCA-Continuous
Implementation of the DCMCA (Deep Centralized Multi Agent Actor Critic) as described in this paper: https://www.researchgate.net/publication/328781547_Managing_engineering_systems_with_large_state_and_action_spaces_through_deep_reinforcement_learning

Adapted for continuous action spaces.

Modifications to original algorithm:
- Target network for ceentral critic
- Adapted for continuous actions for agents/components

Notes:
-Manual tweeking of activation functions for specific problems.

-Implementation error results in soft output clipping, and heavily coordinated suboptimal actions between agents.
