import gymnasium as gym
import shimmy

def build_env(task: str, **kwargs):
    env = gym.make(task, **kwargs)
    # 需要 wrapper 再放這
    return env
