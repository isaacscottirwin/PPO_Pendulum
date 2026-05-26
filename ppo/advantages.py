import torch
class Advantages:
    """
    Computes Generalized Advantage Estimation (GAE) for PPO.
    GAE estimates how much better or worse an action performed compared
    to the critic's expected value estimate.

    It also computes returns:
        return_t = advantage_t + value_t
    which are used as training targets for the critic.

    This class implements:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    and recursively:
        A_t = delta_t + gamma * lambda * A_{t+1}

    where:
        gamma  -> reward discount factor
        lambda -> GAE smoothing parameter

    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize GAE hyperparameters.

        Args:
            gamma:
                Discount factor for future rewards.
                Controls how much the agent values future rewards.

            lam:
                GAE lambda parameter.
                Controls the bias-variance tradeoff
                lambda = 0:
                    low variance, high bias
                lambda = 1:
                    low bias, high variance

        """
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Generalized Advantage Estimation.

        Args:
            rewards:
                Tensor of shape (T,)
                containing rewards collected during rollout.

            values:
                Tensor of shape (T + 1,)
                containing critic predictions for all states plus
                one extra bootstrap value for the final next state.

            dones:
                Tensor of shape (T,)
                indicating whether an episode terminated at timestep t.
                done = 1:
                    episode ended
                done = 0:
                    episode continues

        Returns:
            advantages:
                Tensor of shape (T,)
                containing GAE advantage estimates.

            returns:
                Tensor of shape (T,)
                containing critic training targets.

        """
        # Store computed advantages for each timestep.
        advantages = []

        # Store computed returns for each timestep.
        returns = []

        # Running recursive GAE accumulator.
        #
        # This stores:
        #
        #   delta_t
        # + gamma * lambda * delta_{t+1}
        # + gamma^2 * lambda^2 * delta_{t+2}
        # + ...
        #
        # I compute this backwards through time.
        gae = torch.tensor(0.0, dtype=torch.float32)
        
        # Iterate backwards through the rollout.
        #
        # GAE is recursive:
        #
        #   A_t depends on A_{t+1}
        #
        # so I must compute from the end toward the beginning.

        for t in reversed(range(len(rewards))):
            # Temporal Difference (TD) residual:
            #
            # delta_t =
            #     reward_t
            #   + gamma * V(s_{t+1})
            #   - V(s_t)
            #
            # This measures:
            #
            # "How much better or worse was reality compared
            #  to what the critic predicted?"
            #
            # If done[t] == 1:
            #
            #   gamma * V(s_{t+1}) is removed
            #
            # because there is no future state after termination.
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            # Recursive GAE update:
            #
            # A_t =
            #     delta_t
            #   + gamma * lambda * A_{t+1}
            #
            # Expanded:
            #
            # A_t =
            #     delta_t
            #   + gamma * lambda * delta_{t+1}
            #   + gamma^2 * lambda^2 * delta_{t+2}
            #   + ...
            #
            # This combines multiple future TD errors into a smoother estimate.
            #
            # (1 - dones[t]) stops recursion across episode boundaries.
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae

            # Insert at front because we are iterating backwards.
            advantages.insert(0, gae)
            # Return target for critic:
            #
            # return_t = advantage_t + value_t
            #
            # Rearranging:
            #
            # advantage_t = return_t - value_t
            # so:
            # return_t = advantage_t + value_t
            returns.insert(0, gae + values[t])

        # Convert Python lists into tensors.
        #
        # Final shapes:
        #
        # advantages -> (T,)
        # returns    -> (T,)
        return torch.stack(advantages), torch.stack(returns)