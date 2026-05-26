import gymnasium as gym

def make_env(render_mode=None):
    env = gym.make('Pendulum-v1', render_mode=render_mode)
    return env