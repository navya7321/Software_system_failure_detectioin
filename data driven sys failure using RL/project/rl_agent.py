import os
import numpy as np
from typing import List

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.envs import DummyVecEnv
except Exception:  # pragma: no cover
    PPO = None
    DummyVecEnv = None


class SimpleFailureEnv:
    """A tiny stochastic environment for training without external deps.

    State: [cpu (0..1), mem (0..1), failure_count (0..1)]
    Actions: 0=restart, 1=scale_up, 2=rollback, 3=do_nothing
    Reward: +1 if recovers; -1 otherwise. Episode length = 1 step.
    """

    def __init__(self, seed: int | None = None):
        self.random = np.random.default_rng(seed)
        self.action_space_n = 4
        self.observation_space_shape = (3,)
        self.state = None

    def reset(self):
        self.state = self.random.random(3)
        return self.state

    def step(self, action: int):
        cpu, mem, fail = self.state
        recovered = False
        if action == 0:  # restart
            recovered = (cpu > 0.7 or mem > 0.7) or self.random.random() < 0.6
        elif action == 1:  # scale_up
            recovered = mem > 0.6 or self.random.random() < 0.5
        elif action == 2:  # rollback
            recovered = fail > 0.4 or self.random.random() < 0.4
        else:  # do_nothing
            recovered = self.random.random() < 0.2
        reward = 1.0 if recovered else -1.0
        done = True
        next_state = self.random.random(3)
        self.state = next_state
        return next_state, reward, done, {}


class RLAgent:
    ACTIONS = ['restart', 'scale_up', 'rollback', 'do_nothing']

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.env = SimpleFailureEnv()
        self.model = None
        if PPO is not None and os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path)
            except Exception:
                self.model = None

    def select_action(self, state: List[float]) -> str:
        if self.model is None or PPO is None:
            # Heuristic fallback: choose based on which metric is worse
            cpu, mem, fail = state
            if cpu > mem and cpu > 0.6:
                return 'restart'
            if mem > 0.6:
                return 'scale_up'
            if fail > 0.5:
                return 'rollback'
            return 'do_nothing'
        obs = np.array(state, dtype=np.float32)
        action_idx, _ = self.model.predict(obs, deterministic=True)
        return self.ACTIONS[int(action_idx)]

    def apply_action_effect(self, action: str, cpu: float, mem: float, status: str) -> bool:
        # Simulate recovery likelihood based on action
        if action == 'restart':
            p = 0.7 if status == 'Failed' else 0.5
        elif action == 'scale_up':
            p = 0.6 if mem > cpu else 0.4
        elif action == 'rollback':
            p = 0.5
        else:
            p = 0.2
        return np.random.random() < p

    def train(self, episodes: int = 10):
        if PPO is None:
            return  # Skip if SB3 unavailable
        # Minimal training using SB3 on our simple env surrogate
        class _GymWrapper:
            def __init__(self, env):
                self.env = env
                import gym
                from gym import spaces
                self.action_space = spaces.Discrete(env.action_space_n)
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=env.observation_space_shape, dtype=np.float32)
            def reset(self):
                import numpy as np
                return np.array(self.env.reset(), dtype=np.float32)
            def step(self, action):
                s, r, d, info = self.env.step(int(action))
                import numpy as np
                return np.array(s, dtype=np.float32), float(r), bool(d), info

        gym_env = _GymWrapper(SimpleFailureEnv())
        self.model = PPO('MlpPolicy', gym_env, verbose=0)
        # Track episodic rewards via callback
        from stable_baselines3.common.callbacks import BaseCallback
        self.episode_rewards: list[float] = []
        class RewardLogger(BaseCallback):
            def __init__(self, outer, verbose=0):
                super().__init__(verbose)
                self.outer = outer
                self.ep_reward = 0.0
            def _on_step(self) -> bool:
                # access rewards via infos is env dependent; accumulate via rollout buffer is non-trivial.
                return True
            def _on_rollout_end(self) -> None:
                pass
        self.model.learn(total_timesteps=max(episodes, 1) * 1024, callback=RewardLogger(self))

    def save(self):
        if self.model is not None and PPO is not None:
            self.model.save(self.model_path)
