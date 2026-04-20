import gymnasium as gym
import numpy as np

class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, n: int):
        super().__init__(env)
        self.n = n
        
    def step(self, action):
        total_r = 0.0
        term = trun = False
        info = {}
        
        for _ in range(self.n):
            obs, r, term, trun, info = self.env.step(action)
            total_r += r
            if term or trun:
                break
            
        return obs, total_r, term, trun, info

class PixelObservation(gym.ObservationWrapper):
    ''' obs -> RGB frame '''
    def __init__(self, env: gym.Env, height: int, width: int):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
    
    def observation(self, obs):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                'env.render() return None. Check render_mode="rgb_array" when building env'
            )
        return frame