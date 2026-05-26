import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Combined actor-critic neural network for PPO.

    The actor learns a policy:
        state -> action distribution

    The critic learns a value function:
        state -> expected future return

    For Pendulum-v1:
        obs_dim = 3
        action_dim = 1
        action_scale = 2.0
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_scale: float = 2.0,
    ):
        """
        Initialize the actor network, critic network, and learnable action standard deviation.

        Args:
            obs_dim: Dimension of the observation/state vector.
            action_dim: Dimension of the action vector.
            hidden_dim: Width of the hidden layers.
            action_scale: Maximum absolute action value allowed by the environment.

        """
        super().__init__()
        # Used to scale actor outputs into the environment's valid action range.
        # For Pendulum-v1, the action range is [-2, 2].

        self.action_scale = action_scale

        # Actor network:
        # maps a state to the mean of a Gaussian action distribution.
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Critic network:
        # maps a state to a scalar value estimate V(s).
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable log standard deviation for the Gaussian policy.
        # We store log_std instead of std directly because std must stay positive.
        #
        # exp(-0.5) ≈ 0.61, so the initial policy explores with moderate noise.
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def get_action_and_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return action, log probability, entropy, and value estimate.
        This method is used in two situations:

        1. During rollout collection:
            actions is None, so we sample a new action from the current policy.

        2. During PPO updates:
            actions is provided, so we recompute the log probability of old actions
            under the current policy.

        Args:
            states: State tensor of shape (obs_dim,) or (batch_size, obs_dim).
            actions: Optional action tensor. If None, sample actions from the policy.

        Returns:
            actions: Sampled or provided actions.
            log_probs: Log probability of the actions under the policy.
            entropy: Entropy of the action distribution.
            values: Critic value estimates V(s).
        """
        # Raw actor output is squashed through tanh so the mean lies in [-1, 1],
        # then scaled to the environment action range [-action_scale, action_scale].
        action_mean = torch.tanh(self.actor(states)) * self.action_scale

        # Convert learnable log standard deviation into standard deviation.
        # expand_as(action_mean) makes std match the shape of action_mean,
        # especially for batched states.
        action_std = torch.exp(self.log_std).expand_as(action_mean)

        # Gaussian policy distribution:
        # pi(a | s) = Normal(action_mean, action_std)
        dist = Normal(action_mean, action_std)

        # If no action is supplied, sample one.
        # This is what gives PPO exploration during training.
        if actions is None:
            actions = dist.sample()

        # Compute log probability of the action.
        # sum(dim=-1) collapses across action dimensions.
        # For Pendulum action_dim = 1, but this also works for multi-action envs.
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Entropy measures randomness/exploration of the policy.
        # PPO adds an entropy bonus to discourage premature collapse.
        entropy = dist.entropy().sum(dim=-1)

        # Critic prediction V(s).
        # squeeze(-1) changes shape from (batch_size, 1) to (batch_size,)
        # or from (1,) to scalar-like tensor for single states.
        values = self.critic(states).squeeze(-1)

        return actions, log_probs, entropy, values

    def get_deterministic_action(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the deterministic action from the actor.
        This is used during evaluation, not training.
        Instead of sampling from the Gaussian policy, we use the mean action.
        That makes evaluation less noisy and shows what the learned controller
        actually wants to do.

        Args:
            states: State tensor of shape (obs_dim,) or (batch_size, obs_dim).

        Returns:
            Deterministic action tensor in the valid action range.

        """
        # Same actor mean computation as in get_action_and_value,
        # but without sampling from the Gaussian distribution.
        return torch.tanh(self.actor(states)) * self.action_scale