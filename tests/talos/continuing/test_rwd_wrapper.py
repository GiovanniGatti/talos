import unittest
from typing import Any, Tuple, Dict

import gym
import numpy as np
import pytest

from talos.continuing import rwd_wrapper
from tests.talos.rmenv import tools


class _DummyEnv(gym.Env):

    def __init__(self, rewards: np.ndarray):
        self._rewards = rewards

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        r = self._rewards[self._i]
        self._i += 1
        done = self._i >= self._rewards.shape[0]
        return 0, r, done, {}

    def reset(self):
        self._i = 0


class TestNormalizedReturnWrapper(unittest.TestCase):

    def test_return_is_normalized_according_to_discount_rate(self) -> None:
        env = _DummyEnv(np.ones(20_000))
        env = rwd_wrapper.NormalizedReturnWrapper(env, discount_rate=.99995)

        env.reset()
        collected_rewards = []
        done = False
        while not done:
            _, r, done, _ = env.step(1)
            collected_rewards.append(r)

        assert sum(collected_rewards) == pytest.approx(1., abs=.000001)

    def test_original_return_is_saved_to_infos(self) -> None:
        env = _DummyEnv(np.array([2.]))
        env = rwd_wrapper.NormalizedReturnWrapper(env, discount_rate=.3)

        env.reset()
        _, _, _, infos = env.step(1)

        assert infos['rwd_before_norm'] == 2.


class TestNormalizedRewardWrapper(unittest.TestCase):

    def test_rwd_is_normalized_according_to_optimal_and_random_policies(self) -> None:
        opt_policy = tools.DummyPolicy.any_dummy_policy_state_with(
            horizon=1, initial_capacity=1, vf_mapping=np.array([[0., 0.],
                                                                [0., 3.]]))
        rand_policy = tools.DummyPolicy.any_dummy_policy_state_with(
            horizon=1, initial_capacity=1, vf_mapping=np.array([[0., 0.],
                                                                [0., 1.]]))
        env = _DummyEnv(np.array([2.]))
        env = rwd_wrapper.NormalizedRewardWrapper(env, opt_policy, rand_policy)

        env.reset()
        _, r, _, _ = env.step(1)

        assert r == (2. - 1.) / (3. - 1.)


if __name__ == '__main__':
    unittest.main()
