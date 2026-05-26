import torch


class Losses:
    """
    Stores and computes the PPO loss functions.

    PPO training consists of three major objectives:

    1. Policy loss
        Updates the actor/policy network.

    2. Value loss
        Updates the critic/value network.

    3. Entropy bonus
        Encourages exploration by preventing the policy
        from becoming deterministic too early.

    The final PPO loss is:

        total_loss =
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * entropy_loss
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        Initialize PPO loss hyperparameters.

        Args:
            clip_epsilon:
                PPO clipping threshold.

                Controls how much the policy is allowed
                to change during one update step.

            value_loss_coef:
                Weight applied to critic loss.

            entropy_coef:
                Weight applied to entropy bonus.
                Higher values encourage more exploration.
        """

        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def policy_loss(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PPO clipped policy loss.

        PPO compares:

            new policy probability

        against:

            old policy probability

        to determine how much the policy changed.

        The clipping mechanism prevents excessively large
        policy updates that could destabilize training.

        Args:
            new_logprobs:
                Log probabilities under the CURRENT policy.

            old_logprobs:
                Log probabilities under the OLD policy
                used during rollout collection.

            advantages:
                GAE advantage estimates.

        Returns:
            Scalar PPO policy loss tensor.
        """

        # Compute probability ratio:
        #
        #   pi_new(a|s) / pi_old(a|s)
        #
        # Since we store log probabilities:
        #
        #   exp(new_logprob - old_logprob)
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clip the ratio so the policy cannot
        # change too aggressively in one update.
        #
        # Example:
        #
        #   epsilon = 0.2
        #
        # gives:
        #
        #   ratio ∈ [0.8, 1.2]
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )

        # Standard policy gradient objective.
        #
        # Negative because PyTorch minimizes losses,
        # while PPO mathematically maximizes reward.
        pg_loss1 = -advantages * ratio

        # Clipped PPO objective.
        #
        # Uses clipped ratio to create a more
        # conservative policy update.
        pg_loss2 = -advantages * clipped_ratio

        # PPO takes the WORSE of the two objectives.
        #
        # This creates a conservative lower bound
        # and discourages destructive updates.
        return torch.max(pg_loss1, pg_loss2).mean()

    def value_loss(
        self,
        new_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute critic/value loss.

        The critic learns to predict expected returns.

        Uses Mean Squared Error:

            (V(s) - return)^2

        Args:
            new_values:
                Critic predictions V(s).

            returns:
                Target returns computed using GAE.

        Returns:
            Scalar value loss tensor.
        """

        # Mean Squared Error between
        # predicted values and target returns.
        return ((new_values - returns) ** 2).mean()

    def entropy_loss(
        self,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy bonus.

        Entropy measures randomness of the policy.

        Higher entropy:
            more exploration

        Lower entropy:
            more deterministic behavior

        PPO subtracts entropy from total loss
        to encourage exploration.

        Args:
            entropy:
                Entropy values from the action distribution.

        Returns:
            Mean entropy scalar tensor.
        """

        # Average entropy across batch.
        return entropy.mean()

    def total_loss(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total PPO loss.

        Final PPO objective:

            total_loss =
                policy_loss
                + value_loss_coef * value_loss
                - entropy_coef * entropy_loss

        Args:
            policy_loss:
                PPO clipped policy loss.

            value_loss:
                Critic/value loss.

            entropy_loss:
                Entropy bonus.

        Returns:
            Scalar total loss tensor.
        """

        # Combine PPO objectives into one scalar loss.
        #
        # Entropy is SUBTRACTED because:
        #
        #   higher entropy should reduce loss,
        #
        # encouraging exploration.
        return (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy_loss
        )