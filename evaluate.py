# evaluate.py

import gymnasium as gym
import torch

from models.actor_critic import ActorCritic


def main():
    """
    Load a trained PPO policy and visualize it interacting
    with the Pendulum-v1 environment.

    This script:
        1. Creates the environment.
        2. Loads trained model weights.
        3. Runs the policy in evaluation mode.
        4. Renders the environment using pygame.
    """

    # Create Pendulum environment with rendering enabled.
    #
    # render_mode="human":
    #     opens a pygame window so we can visually
    #     observe the trained policy.
    env = gym.make(
        "Pendulum-v1",
        render_mode="human",
    )

    # Observation space dimension.
    #
    # Pendulum observations:
    #   [cos(theta), sin(theta), angular_velocity]
    #
    # Shape:
    #   (3,)
    obs_dim = env.observation_space.shape[0]

    # Action space dimension.
    #
    # Pendulum has a single continuous torque action.
    #
    # Shape:
    #   (1,)
    action_dim = env.action_space.shape[0]

    # Create actor-critic model.
    #
    # Architecture must EXACTLY match the architecture
    # used during training or checkpoint loading will fail.
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Path to the best checkpoint saved during training.
    checkpoint_path = "checkpoints/ppo_pendulum_best.pt"

    # Load trained model weights.
    #
    # map_location:
    #     ensures checkpoint loads onto CPU.
    #
    # weights_only=True:
    #     loads only parameter tensors.
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )

    print(f"Loaded {checkpoint_path}")

    # Put model into evaluation mode.
    #
    # Important for networks using:
    #   - dropout
    #   - batch normalization
    #
    # (not strictly necessary here but still best practice)
    model.eval()

    # Reset environment and get initial state.
    state, info = env.reset()

    # Main evaluation loop.
    while True:

        # Convert NumPy state array into PyTorch tensor.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        )

        # Disable gradient tracking during evaluation.
        #
        # This reduces memory usage and speeds up inference.
        with torch.no_grad():

            # Use deterministic action selection.
            #
            # During evaluation we usually want the mean action
            # rather than stochastic exploration noise.
            action = model.get_deterministic_action(
                state_tensor
            )

        # Step environment forward using chosen action.
        next_state, reward, terminated, truncated, info = env.step(
            action.numpy()
        )

        # Determine if episode ended.
        #
        # terminated:
        #     natural episode termination
        #
        # truncated:
        #     time limit reached
        done = terminated or truncated

        # Move to next environment state.
        state = next_state

        # If episode ends:
        #   reset environment once
        #   then exit program
        if done:
            state, info = env.reset()
            break

    # Close pygame window and cleanup environment resources.
    env.close()


# Standard Python entry point.
#
# Ensures main() only runs when this file
# is executed directly.
if __name__ == "__main__":
    main()