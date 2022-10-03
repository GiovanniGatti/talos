import random
from typing import Tuple, Optional, Any, Dict

import gym
import numpy as np
from gym import spaces

from talos import rmenv


class DummyPolicy(rmenv.PolicyState):

    @classmethod
    def any_dummy_policy_state_with(cls,
                                    horizon: Optional[int] = random.randint(1, 365),
                                    initial_capacity: Optional[int] = random.randint(1, 365),
                                    vf_mapping: Optional[np.ndarray] = np.random.random((3, 4)),
                                    policy_mapping: Optional[np.ndarray] = np.random.random((3, 4))) -> 'DummyPolicy':
        return DummyPolicy(horizon, initial_capacity, vf_mapping, policy_mapping)

    def __init__(self, horizon: int, initial_capacity: int, vf_mapping: np.ndarray, policy_mapping: np.ndarray):
        self._horizon = horizon
        self._initial_capacity = initial_capacity
        self._vf_mapping = vf_mapping
        self._policy_mapping = policy_mapping

    def horizon(self) -> int:
        return self._horizon

    def initial_cap(self) -> int:
        return self._initial_capacity

    def v(self, dtd: int, cap: int) -> float:
        return self._vf_mapping[dtd, cap]

    def pi(self, dtd: int, cap: int) -> float:
        return self._policy_mapping[dtd, cap]

    def __getitem__(self, item: Tuple[np.ndarray, ...]) -> np.ndarray:
        return self._policy_mapping[item]

    def update(self, new_arrival_rate: float, new_price_sensitivity: float) -> None:
        raise NotImplementedError


class DummyEnv(gym.Env):

    def __init__(self, action_space: gym.Space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)):
        self.action_space = action_space
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(1,), dtype=int),))

    def step(self, action: float) -> Tuple[Tuple[Any], float, bool, Dict[Any, Any]]:
        return (0,), 0., True, {}

    def reset(self) -> Tuple[Any]:
        return 0,

    def render(self, mode: str = 'human') -> None:
        pass
