import itertools
import math
import random
import unittest
from collections import defaultdict
from typing import Callable, Dict, Tuple, Any, Iterable

import numpy as np
import pytest
from scipy.stats import poisson

from talos import rmenv
from talos.continuing import env
from tests.talos import continuing
from tests.talos.rmenv import tools


# noinspection Mypy
# lambda function returns boolean
def sample_until_no_resources_have_exhausted_capacity(e: env.ContinuingSingleLeg) -> np.ndarray:
    return sample_until(e, lambda s: (s[:, 1] > 0).all())


# noinspection Mypy
# lambda function returns boolean
def sample_until_at_least_one_resource_has_exhausted_capacity(e: env.ContinuingSingleLeg) -> np.ndarray:
    return sample_until(e, lambda s: (s[:, 1] == 0).any())


def sample_until(e: env.ContinuingSingleLeg, condition: Callable[[np.ndarray], bool]) -> np.ndarray:
    while True:
        s, _, _, _ = e.step(np.random.randint(0, 1, size=2))
        if condition(s):
            return s


class _CapturingDummyEnv(tools.DummyEnv):

    def __init__(self, obs: Iterable[Any]):
        super().__init__()
        self.selected_actions = []
        self._obs = iter(obs)

    def step(self, action: np.ndarray) -> Tuple[Tuple[Any], float, bool, Dict[Any, Any]]:
        self.selected_actions.append(action)
        return next(self._obs), 0., False, {}

    def reset(self) -> Tuple[Any]:
        return next(self._obs)


class TestContinuingSingleLeg(unittest.TestCase):

    def test_follows_the_rwd_fn_contract(self) -> None:
        _rwd_fn = continuing.CapturingRwdFn(rwd=25.)
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        _env = continuing.any_continuing_single_leg_env_with(param_state, horizon=13, _rwd_fn=_rwd_fn)

        s = _env.reset()
        a = np.random.randint(0, 1, size=13)
        ns, r, _, _ = _env.step(a)

        assert (_rwd_fn.cap == s[:, 1]).all()
        assert (_rwd_fn.selected_fares == a).all()
        assert np.all(s[1:, 1] - ns[:-1, 1] == _rwd_fn.bookings[1:])
        assert r == 25.

    def test_env_demand_generation_follows_poisson_and_exp_decay_wtp(self) -> None:
        _rwd_fn = continuing.CapturingRwdFn()
        param_state = rmenv.DemandParameterState(arrival_rate=.4, price_sensitivity=.7)
        _env = continuing.any_continuing_single_leg_env_with(param_state,
                                                             horizon=1,
                                                             initial_capacity=100,
                                                             fare_structure=np.array([50., 70.]),
                                                             _rwd_fn=_rwd_fn)

        data: Dict[int, float] = defaultdict(float)
        _env.reset()
        for _ in range(50_000):
            _env.step(np.array([1]))
            data[int(_rwd_fn.bookings.item())] += 1

        assert data
        assert sum(data.values()) == 50_000
        # testing only the most likely outcomes,
        # otherwise it will require lots of samples to obtain an acceptable margin error
        _mu = (.4 * math.exp(-.7 * (70. / 50. - 1)))
        for k in range(3):
            assert data[k] / 50_000 == pytest.approx(poisson.pmf(k=k, mu=_mu), rel=.05)

    def test_info_reports_how_many_optimal_actions_were_chosen(self) -> None:
        # high arrival rate, low capacity, low price sensitivity produces an optimal policy which optimal fare is $70
        _rwd_fn = continuing.CapturingRwdFn()
        param_state = rmenv.DemandParameterState(arrival_rate=.9, price_sensitivity=.5)
        _env = continuing.any_continuing_single_leg_env_with(param_state,
                                                             horizon=2,
                                                             initial_capacity=2,
                                                             fare_structure=np.array([50., 70.]),
                                                             _rwd_fn=_rwd_fn)

        _env.reset()

        sample_until_no_resources_have_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([0, 0]))
        assert info['optimal_action'] == 0.

        sample_until_no_resources_have_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([1, 0]))
        assert info['optimal_action'] == .5

        sample_until_no_resources_have_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([0, 1]))
        assert info['optimal_action'] == .5

        sample_until_no_resources_have_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([1, 1]))
        assert info['optimal_action'] == 1.

    def test_info_optimal_actions_counts_only_non_exhausted_resources(self) -> None:
        # high arrival rate, low capacity, low price sensitivity produces an optimal policy which optimal fare is $70
        _rwd_fn = continuing.CapturingRwdFn()
        param_state = rmenv.DemandParameterState(arrival_rate=.9, price_sensitivity=.5)
        _env = continuing.any_continuing_single_leg_env_with(param_state,
                                                             horizon=2,
                                                             initial_capacity=2,
                                                             fare_structure=np.array([50., 70.]),
                                                             _rwd_fn=_rwd_fn)

        _env.reset()

        sample_until_at_least_one_resource_has_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([0, 0]))
        assert info['optimal_action'] == 0.

        sample_until_at_least_one_resource_has_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([1, 0]))
        assert info['optimal_action'] == 0.

        sample_until_at_least_one_resource_has_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([0, 1]))
        assert info['optimal_action'] == 1.

        sample_until_at_least_one_resource_has_exhausted_capacity(_env)
        _, _, _, info = _env.step(np.array([1, 1]))
        assert info['optimal_action'] == 1.

    def test_info_returns_the_raw_revenue(self) -> None:
        _rwd_fn = continuing.CapturingRwdFn()
        param_state = rmenv.DemandParameterState(arrival_rate=.9, price_sensitivity=.5)
        _env = continuing.any_continuing_single_leg_env_with(param_state,
                                                             horizon=2,
                                                             initial_capacity=2,
                                                             fare_structure=np.array([50., 70.]),
                                                             _rwd_fn=_rwd_fn)

        _env.reset()
        _, _, _, info = _env.step(np.array([1, 0]))
        assert info['raw_rev'] == np.sum(_rwd_fn.bookings * np.array([70., 50.]))

    def test_env_manages_inventory_correctly_and_it_emits_valid_states(self) -> None:
        _rwd_fn = continuing.CapturingRwdFn()
        param_state = rmenv.DemandParameterState(arrival_rate=.9, price_sensitivity=.5)
        _env = continuing.any_continuing_single_leg_env_with(param_state,
                                                             horizon=2,
                                                             initial_capacity=2,
                                                             fare_structure=np.array([50., 70.]),
                                                             _rwd_fn=_rwd_fn)

        prev_state = _env.reset()
        for _ in range(1000):
            state, _, done, _ = _env.step(np.array([0, 0]))
            expected_state = np.roll(prev_state[:, 1] - _rwd_fn.bookings, 1)
            expected_state[-1] = 2
            expected_state = np.column_stack((prev_state[:, 0], expected_state))

            assert not done
            assert (state == expected_state).all()
            assert _env.observation_space.contains(state)
            prev_state = state


class TestWarmupWrapper(unittest.TestCase):

    def test_warmup_policy_is_executed_during_reset(self) -> None:
        _policy_state = tools.DummyPolicy.any_dummy_policy_state_with(policy_mapping=np.array([[0, 1, 2],
                                                                                               [3, 4, 5],
                                                                                               [6, 7, 8],
                                                                                               [9, 10, 11]]))
        _env = _CapturingDummyEnv(itertools.repeat(np.array([[0, 0],
                                                             [1, 1],
                                                             [2, 2]])))
        wrapped = env.WarmupWrapper(_env, _policy_state, horizon=1)

        obs = wrapped.reset()

        assert np.alltrue(obs == np.array([[0, 0], [1, 1], [2, 2]]))
        assert np.alltrue(np.array(_env.selected_actions) == np.array([[3, 7, 11]]))  # dtd=0 is ignored

    def test_warmup_until_horizon(self) -> None:
        any_policy = np.random.randint(0, 100, size=15).reshape((5, 3))
        _policy_state = tools.DummyPolicy.any_dummy_policy_state_with(policy_mapping=any_policy)
        any_states = np.array_split(np.stack((np.random.randint(0, 4, size=3 * 24),
                                              np.random.randint(0, 3, size=3 * 24))).T,
                                    24)
        _env = _CapturingDummyEnv(any_states)
        wrapped = env.WarmupWrapper(_env, _policy_state, horizon=23)

        wrapped.reset()

        assert len(_env.selected_actions) == 23


if __name__ == '__main__':
    unittest.main()
