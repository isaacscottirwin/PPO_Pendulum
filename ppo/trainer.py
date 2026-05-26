import torch

from ppo.buffer import Buffer
from ppo.losses import Losses


class PPOTrainer:
    """
    Handles PPO training updates.

    PPO training occurs in two major phases:

        1. Collect rollout data from the environment.
        2. Repeatedly update the policy using minibatches
           sampled from that rollout.

    This class performs the gradient update phase.

    Responsibilities:
        - shuffle rollout data
        - create minibatches
        - compute PPO losses
        - perform gradient descent
        - track training metrics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: Losses,
        update_epochs: int,
        minibatch_size: int,
        max_grad_norm: float | None = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            model:
                Actor-critic neural network.

            optimizer:
                Optimizer used for gradient descent.

            losses:
                PPO loss helper class.

            update_epochs:
                Number of passes over rollout data.

            minibatch_size:
                Number of samples per minibatch.

            max_grad_norm:
                Optional gradient clipping threshold.
        """

        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

    def update(self, batch: Buffer) -> dict[str, float]:
        """
        Perform PPO updates using rollout data.

        PPO does NOT update after every environment step.

        Instead:
            - collect a large rollout
            - repeatedly sample minibatches
            - update the policy several times

        Args:
            batch:
                Buffer containing rollout data.

        Returns:
            Dictionary containing averaged training metrics.
        """

        # PPO requires precomputed advantages.
        #
        # Advantages measure:
        #
        #   how much better/worse an action performed
        #
        # compared to the critic's expectation.
        if batch.advantages is None:
            raise ValueError(
                "batch.advantages is None. Please compute advantages before calling update."
            )

        # PPO also requires return targets
        # for critic regression.
        if batch.returns is None:
            raise ValueError(
                "batch.returns is None. Please compute returns before calling update."
            )

        # Number of rollout samples.
        #
        # Example:
        #   2048
        batch_size = batch.states.shape[0]

        # Normalize advantages.
        #
        # This stabilizes PPO training by preventing
        # very large advantage values from dominating
        # gradient updates.
        advantages = batch.advantages.detach()

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Create detached training batch.
        #
        # We detach rollout tensors because rollout collection
        # should not remain connected to the computation graph.
        #
        # PPO treats rollout data as fixed during updates.
        batch = Buffer(
            states=batch.states.detach(),
            actions=batch.actions.detach(),
            rewards=batch.rewards.detach(),
            dones=batch.dones.detach(),
            logprobs=batch.logprobs.detach(),
            values=batch.values.detach(),
            advantages=advantages,
            returns=batch.returns.detach(),
        )

        # Store cumulative metrics for logging.
        #
        # These values are averaged at the end.
        loss_sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }

        # Count total minibatch updates performed.
        num_updates = 0

        # PPO performs multiple passes over the same rollout.
        #
        # Example:
        #   update_epochs = 10
        #
        # This improves sample efficiency.
        for _ in range(self.update_epochs):

            # Randomly shuffle rollout indices.
            #
            # PPO uses random minibatches rather than
            # sequential slices.
            indices = torch.randperm(batch_size)

            # Iterate through minibatches.
            for start in range(0, batch_size, self.minibatch_size):

                end = start + self.minibatch_size

                # Select minibatch indices.
                mb_indices = indices[start:end]

                # Create minibatch buffer.
                mb = batch.select(mb_indices)

                # Run actor-critic network.
                #
                # This computes:
                #
                #   - new action log probabilities
                #   - entropy
                #   - updated critic values
                #
                # using the CURRENT policy.
                _, new_logprobs, entropy, new_values = (
                    self.model.get_action_and_value(
                        mb.states,
                        mb.actions,
                    )
                )

                # Compute PPO importance sampling ratio.
                #
                # ratio =
                #
                #   pi_new(a|s) / pi_old(a|s)
                #
                # PPO uses this to measure how much
                # the policy changed.
                log_ratio = new_logprobs - mb.logprobs

                ratio = torch.exp(log_ratio)

                # Compute logging diagnostics.
                #
                # approx_kl:
                #   approximate KL divergence between policies
                #
                # clipfrac:
                #   fraction of samples affected by PPO clipping
                with torch.no_grad():

                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                    clipfrac = (
                        ((ratio - 1.0).abs() > self.losses.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )

                # Compute PPO clipped policy loss.
                policy_loss = self.losses.policy_loss(
                    new_logprobs,
                    mb.logprobs,
                    mb.advantages,
                )

                # Compute critic regression loss.
                #
                # Critic learns:
                #
                #   V(s) ≈ return
                value_loss = self.losses.value_loss(
                    new_values,
                    mb.returns,
                )

                # Compute entropy bonus.
                #
                # Encourages exploration.
                entropy_loss = self.losses.entropy_loss(entropy)

                # Combine PPO losses into final loss.
                loss = self.losses.total_loss(
                    policy_loss,
                    value_loss,
                    entropy_loss,
                )

                # Clear previous gradients.
                self.optimizer.zero_grad()

                # Backpropagate gradients.
                loss.backward()

                # Optional gradient clipping.
                #
                # Prevents exploding gradients and stabilizes training.
                if self.max_grad_norm is not None:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                # Apply optimizer step.
                self.optimizer.step()

                # Accumulate metrics for averaging later.
                loss_sums["loss"] += loss.item()
                loss_sums["policy_loss"] += policy_loss.item()
                loss_sums["value_loss"] += value_loss.item()
                loss_sums["entropy_loss"] += entropy_loss.item()
                loss_sums["approx_kl"] += approx_kl
                loss_sums["clipfrac"] += clipfrac

                num_updates += 1

        # Return average metrics across all minibatch updates.
        return {
            key: value / num_updates
            for key, value in loss_sums.items()
        }