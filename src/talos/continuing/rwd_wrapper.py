from typing import Any, Dict, Tuple

import gym

from talos import rmenv


class NormalizedReturnWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, discount_rate: float):
        super(NormalizedReturnWrapper, self).__init__(env)
        self._discount_rate_scale = 1. / (1 - discount_rate) if discount_rate < 1. else 1.

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        ns, r, done, infos = super().step(action)
        assert 'rwd_before_norm' not in infos
        infos['rwd_before_norm'] = r
        return ns, r / self._discount_rate_scale, done, infos


class NormalizedRewardWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, opt_policy_state: rmenv.PolicyState, random_policy_state: rmenv.PolicyState):
        super(NormalizedRewardWrapper, self).__init__(env)

        assert opt_policy_state.horizon() == random_policy_state.horizon()
        assert opt_policy_state.initial_cap() == random_policy_state.initial_cap()

        self._opt_policy_state = opt_policy_state
        self._random_policy_state = random_policy_state
        self._cap = opt_policy_state.initial_cap()
        self._horizon = opt_policy_state.horizon()

    def reward(self, reward: float) -> float:
        return (reward - self._offset) / (self._scale - self._offset)

    def reset(self, **kwargs: Dict[str, Any]) -> Any:
        s = super().reset(**kwargs)
        self._scale = self._opt_policy_state.v(self._horizon, self._cap)
        self._offset = self._random_policy_state.v(self._horizon, self._cap)
        return s
