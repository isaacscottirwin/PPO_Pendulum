from dataclasses import dataclass


@dataclass
class PPOConfig:
    LEARNING_RATE: float = 3e-4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_COEF: float = 0.2

    ROLLOUT_STEPS: int = 2048
    UPDATE_EPOCHS: int = 10
    MINIBATCH_SIZE: int = 64

    ENTROPY_COEF: float = 0.01
    VALUE_COEF: float = 0.5

    TOTAL_ITERATIONS: int = 1000
    SEED: int = 1