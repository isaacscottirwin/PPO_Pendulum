from dataclasses import dataclass
import torch


@dataclass
class Buffer:
    """
    Stores rollout data collected from the environment.

    PPO first interacts with the environment and collects trajectories:

        state -> action -> reward -> next_state

    over many timesteps.

    After rollout collection, PPO repeatedly samples minibatches
    from this buffer to perform gradient updates.

    This class stores:
        - states
        - actions
        - rewards
        - done flags
        - old log probabilities
        - critic value estimates
        - advantages
        - returns
    """

    # Environment observations/states.
    #
    # Shape:
    #   (T, obs_dim)
    #
    # Example:
    #   (2048, 3) for Pendulum-v1
    states: torch.Tensor

    # Actions sampled from the policy.
    #
    # Shape:
    #   (T, action_dim)
    #
    # Example:
    #   (2048, 1)
    actions: torch.Tensor

    # Rewards received after each action.
    #
    # Shape:
    #   (T,)
    rewards: torch.Tensor

    # Episode termination indicators.
    #
    # done = 1:
    #   episode ended
    #
    # done = 0:
    #   episode continues
    #
    # Shape:
    #   (T,)
    dones: torch.Tensor

    # Log probabilities of actions under the OLD policy.
    #
    # PPO compares:
    #
    #   old_logprobs
    #
    # against:
    #
    #   new_logprobs
    #
    # to compute the policy ratio:
    #
    #   pi_new(a|s) / pi_old(a|s)
    #
    # Shape:
    #   (T,)
    logprobs: torch.Tensor

    # Critic value estimates V(s).
    #
    # Shape:
    #   (T,)
    values: torch.Tensor

    # GAE advantage estimates.
    #
    # Measures how much better or worse
    # an action performed than expected.
    #
    # Shape:
    #   (T,)
    #
    # Initially None until computed after rollout collection.
    advantages: torch.Tensor | None = None

    # Return targets for critic training.
    #
    # Shape:
    #   (T,)
    #
    # Initially None until computed after rollout collection.
    returns: torch.Tensor | None = None

    def select(
        self,
        indices: torch.Tensor,
    ) -> "Buffer":
        """
        Create a minibatch buffer using selected indices.

        PPO randomly samples minibatches from the rollout
        during gradient updates.

        Instead of training on the entire rollout at once,
        PPO repeatedly trains on smaller shuffled subsets.

        Args:
            indices:
                Tensor containing minibatch indices.

        Returns:
            A smaller Buffer containing only the selected samples.
        """

        return Buffer(

            # Select minibatch states.
            states=self.states[indices],

            # Select minibatch actions.
            actions=self.actions[indices],

            # Select minibatch rewards.
            rewards=self.rewards[indices],

            # Select minibatch done flags.
            dones=self.dones[indices],

            # Select minibatch old log probabilities.
            logprobs=self.logprobs[indices],

            # Select minibatch critic values.
            values=self.values[indices],

            # Select minibatch advantages if they exist.
            advantages=(
                self.advantages[indices]
                if self.advantages is not None
                else None
            ),

            # Select minibatch returns if they exist.
            returns=(
                self.returns[indices]
                if self.returns is not None
                else None
            ),
        )