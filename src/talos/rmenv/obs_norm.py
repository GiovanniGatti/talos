import gym
import numpy as np
from gym import spaces


class MinMax(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_obs_space = env.observation_space

        if not isinstance(base_obs_space, spaces.MultiDiscrete):
            raise ValueError('Only MultiDiscrete spaces are currently supported, but found {}'.format(base_obs_space))

        self._max = base_obs_space.nvec - 1
        self.observation_space = spaces.Box(low=-1., high=1., shape=base_obs_space.shape, dtype=float)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return 2 * (observation / self._max) - 1.


class OneHot(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_obs_space = env.observation_space

        if not isinstance(base_obs_space, spaces.MultiDiscrete):
            raise ValueError('Expected state space of type MultiDiscrete, but found {}'.format(base_obs_space))

        if base_obs_space.nvec.ndim != 2:
            raise ValueError('Unsupported MultiDiscrete spec. Expected 2 dim obs. space, but found {}'
                             .format(base_obs_space.nvec))

        self._column = tuple(np.squeeze(np.unique(base_obs_space.nvec, axis=0)))
        self._shape = (base_obs_space.shape[0], np.sum(np.unique(base_obs_space.nvec, axis=0)))
        self.observation_space = spaces.Box(low=0., high=1., shape=self._shape, dtype=float)
        self._row_indexes = np.arange(self._shape[0])

    def observation(self, observation: np.ndarray) -> np.ndarray:
        arr = np.zeros(self._shape)
        k = 0
        for i, j in enumerate(self._column):
            arr[self._row_indexes, observation[:, i] + k] = 1.
            k += j
        return arr
