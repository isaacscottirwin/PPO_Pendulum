import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from configs.ppo_config import PPOConfig
from models.actor_critic import ActorCritic
from ppo.advantages import Advantages
from ppo.buffer import Buffer
from ppo.losses import Losses
from ppo.trainer import PPOTrainer


def evaluate_policy(env, model, episodes=5):
    """
    Evaluate the current policy using deterministic actions.

    During training the policy samples actions stochastically.
    During evaluation we remove exploration noise and use
    the deterministic policy output instead.
    """

    total_rewards = []

    for _ in range(episodes):
        state, info = env.reset()

        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            # Disable gradient tracking during evaluation.
            with torch.no_grad():
                action = model.get_deterministic_action(
                    state_tensor
                )

            state, reward, terminated, truncated, info = env.step(
                action.numpy()
            )

            done = terminated or truncated

            episode_reward += reward

        total_rewards.append(episode_reward)

    return sum(total_rewards) / len(total_rewards)


def plot_training_curve(
    reward_history,
    window=25,
):
    """
    Plot PPO training rewards and save the figure.

    Args:
        reward_history:
            List containing rollout rewards.

        window:
            Moving average window size.
    """

    # Create visualization directory if needed.
    os.makedirs("visualization", exist_ok=True)

    # Compute moving average for smoother visualization.
    moving_avg = [
        sum(reward_history[i - window : i]) / window
        for i in range(window, len(reward_history))
    ]

    # Plot raw reward history.
    plt.plot(
        reward_history,
        alpha=0.3,
        label="Raw Reward",
    )

    # Plot moving average.
    plt.plot(
        range(window, len(reward_history)),
        moving_avg,
        label="Moving Average",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Total Rollout Reward")
    plt.title("PPO Training Progress")

    plt.legend()

    # Save plot to visualization folder.
    plt.savefig("visualization/ppo_training_curve.png")

    # Display plot window.
    plt.show()


config = PPOConfig()

# Set random seeds for reproducibility.
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# Create Pendulum environment.
env = gym.make("Pendulum-v1")

# Observation dimension:
#
# Pendulum-v1:
#   [cos(theta), sin(theta), theta_dot]
obs_dim = env.observation_space.shape[0]

# Action dimension:
#
# Pendulum-v1 has one torque action.
action_dim = env.action_space.shape[0]

# Create actor-critic network.
model = ActorCritic(obs_dim, action_dim)

# Adam optimizer for gradient descent.
optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
)

# Generalized Advantage Estimation helper.
advantages_calculator = Advantages(
    gamma=config.GAMMA,
    lam=config.GAE_LAMBDA,
)

# PPO loss helper.
losses = Losses(
    clip_epsilon=config.CLIP_COEF,
    value_loss_coef=config.VALUE_COEF,
    entropy_coef=config.ENTROPY_COEF,
)

# PPO trainer object.
trainer = PPOTrainer(
    model=model,
    optimizer=optimizer,
    losses=losses,
    update_epochs=config.UPDATE_EPOCHS,
    minibatch_size=config.MINIBATCH_SIZE,
)

# Create checkpoint directory.
os.makedirs("checkpoints", exist_ok=True)

best_eval_reward = float("-inf")

best_checkpoint_path = "checkpoints/ppo_pendulum_best.pt"
final_checkpoint_path = "checkpoints/ppo_pendulum.pt"

# Reset environment and set seed.
state, info = env.reset(seed=config.SEED)

# Track training rewards for plotting.
reward_history = []

# Main PPO training loop.
for iteration in range(config.TOTAL_ITERATIONS):

    # Rollout storage.
    states = []
    actions = []
    rewards = []
    dones = []
    logprobs = []
    values = []

    total_reward = 0

    # Collect rollout trajectory.
    for step in range(config.ROLLOUT_STEPS):

        # Convert numpy state into tensor.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        )

        # Disable gradients during rollout collection.
        with torch.no_grad():

            # Sample action from current policy.
            #
            # Returns:
            #   action
            #   log probability
            #   entropy
            #   critic value
            action, logprob, _, value = model.get_action_and_value(
                state_tensor
            )

        # Pendulum expects actions in [-2, 2].
        env_action = torch.clamp(
            action,
            -model.action_scale,
            model.action_scale,
        )

        # Step environment forward.
        next_state, reward, terminated, truncated, info = env.step(
            env_action.numpy()
        )

        done = terminated or truncated

        # Store rollout transition.
        states.append(state_tensor)
        actions.append(action)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        dones.append(torch.tensor(done, dtype=torch.float32))
        logprobs.append(logprob)
        values.append(value)

        total_reward += reward

        # Advance environment state.
        state = next_state

        # Reset environment if episode ended.
        if done:
            state, info = env.reset()

    # Compute bootstrap value for final state.
    #
    # GAE requires:
    #
    #   V(s_t+1)
    with torch.no_grad():
        final_state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        )

        final_value = model.critic(
            final_state_tensor
        ).squeeze()

    values.append(final_value)

    # Convert rollout lists into tensors.
    rewards_tensor = torch.stack(rewards)
    dones_tensor = torch.stack(dones)
    values_tensor = torch.stack(values)

    # Compute GAE advantages and return targets.
    advantages_tensor, returns_tensor = (
        advantages_calculator.compute_advantages(
            rewards=rewards_tensor,
            values=values_tensor,
            dones=dones_tensor,
        )
    )

    # Create PPO training batch.
    batch = Buffer(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=rewards_tensor,
        dones=dones_tensor,
        logprobs=torch.stack(logprobs),
        values=values_tensor[:-1],
        advantages=advantages_tensor,
        returns=returns_tensor,
    )

    # Perform PPO updates.
    metrics = trainer.update(batch)

    reward_history.append(total_reward)

    # Periodically evaluate policy.
    if iteration % 25 == 0:

        eval_reward = evaluate_policy(
            env,
            model,
        )

        saved_best = ""

        # Save best-performing model.
        if eval_reward > best_eval_reward:

            best_eval_reward = eval_reward

            torch.save(
                model.state_dict(),
                best_checkpoint_path,
            )

            saved_best = " [new best, saved]"

        # Print training metrics.
        print(
            f"Iteration {iteration} | "
            f"Train Reward: {total_reward:.2f} | "
            f"Eval Reward: {eval_reward:.2f} | "
            f"Loss: {metrics['loss']:.4f}"
            f"{saved_best}"
        )

# Save final model weights.
torch.save(
    model.state_dict(),
    final_checkpoint_path,
)

# Plot and save training curve.
plot_training_curve(reward_history)

print(
    f"Best eval reward: "
    f"{best_eval_reward:.2f} "
    f"-> {best_checkpoint_path}"
)

print(
    f"Final weights saved -> "
    f"{final_checkpoint_path}"
)

# Close environment.
env.close()