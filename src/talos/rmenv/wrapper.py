from typing import List, Any, Tuple, Dict

import gym
import numpy as np
from gym import spaces

from talos import rmenv


class SampleParameterWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, _state: rmenv.DemandParameterState, _sampler: rmenv.Sampler,
                 _policy_states: List[rmenv.PolicyState]):
        super().__init__(env)
        self._state = _state
        self._sampler = _sampler
        self._policy_states = _policy_states

    def reset(self) -> np.ndarray:
        arrival_rate, price_sensitivity = self._sampler.sample()
        self._state.arrival_rate = arrival_rate
        self._state.price_sensitivity = price_sensitivity
        for state in self._policy_states:
            state.update(arrival_rate, price_sensitivity)
        return super().reset()


class TrueParamsObsWrapper(gym.ObservationWrapper):

    def __init__(self,
                 base_env: gym.Env,
                 state: rmenv.DemandParameterState,
                 params_range: List[Tuple[float, float]]):
        super(TrueParamsObsWrapper, self).__init__(base_env)
        self._base_env = base_env
        self._state = state
        self._params_range = np.array(params_range)
        self._mask = self._params_range[:, 0] != self._params_range[:, 1]
        self._shape = np.sum(self._mask)
        self._min = np.min(self._params_range[self._mask], axis=1)
        self._max = np.max(self._params_range[self._mask], axis=1)
        if self._shape > 0:
            self.observation_space = spaces.Tuple((base_env.observation_space,
                                                   spaces.Box(low=-1., high=1., shape=(self._shape,), dtype=float)))

    def observation(self, observation: Any) -> Tuple[Any, np.ndarray]:
        if self._shape == 0:
            return observation

        params = np.array([self._state.arrival_rate, self._state.price_sensitivity])
        params = params[self._mask]
        params = 2 * (params - self._min) / (self._max - self._min) - 1
        return observation, params

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        o, r, done, infos = super().step(action)
        if 'orig_obs' in infos:
            infos['orig_obs'] = infos['orig_obs'], self._state.arrival_rate, self._state.price_sensitivity
        return o, r, done, infos
