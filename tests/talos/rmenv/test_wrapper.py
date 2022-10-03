import itertools
import random
import unittest
from typing import List, Tuple

import numpy as np
from gym import spaces

from talos import rmenv
from tests.talos.rmenv import tools


class _CycleSampler(rmenv.Sampler):

    def __init__(self, parameters: List[Tuple[float, float]]):
        self._parameters = itertools.cycle(parameters)

    def sample(self) -> Tuple[float, float]:
        return next(self._parameters)


class _CapturingPolicyState(rmenv.PolicyState):

    def __init__(self) -> None:
        self.updated_params: List[Tuple[float, float]] = []

    def horizon(self) -> int:
        raise NotImplementedError

    def initial_cap(self) -> int:
        raise NotImplementedError

    def v(self, dtd: int, cap: int) -> float:
        raise NotImplementedError

    def pi(self, dtd: int, cap: int) -> float:
        raise NotImplementedError

    def __getitem__(self, item: Tuple[np.ndarray, ...]) -> np.ndarray:
        raise NotImplementedError

    def update(self, new_arrival_rate: float, new_price_sensitivity: float) -> None:
        self.updated_params.append((new_arrival_rate, new_price_sensitivity))


class TestSampleParameterWrapper(unittest.TestCase):

    def test_policy_states_are_updated_after_each_new_sampled_set_of_parameters(self) -> None:
        _sampler = _CycleSampler(parameters=[(1., 2.), (3., 4.)])
        _state = rmenv.DemandParameterState(1., 2.)
        _policy_state = _CapturingPolicyState()
        wrapped = rmenv.SampleParameterWrapper(tools.DummyEnv(), _state, _sampler, _policy_states=[_policy_state])

        wrapped.reset()  # first reset, update to (1., 2.)
        wrapped.reset()  # second reset, update to (3., 4.)
        wrapped.reset()  # third reset, update back to (1., 2.)

        assert _policy_state.updated_params == [(1., 2.), (3., 4.), (1., 2.)]


class TestPriceSensitivityObsWrapper(unittest.TestCase):

    def test_step_observations_are_wrapped_with_normalized_price_sensitivity(self) -> None:
        any_arrival_rate = random.randint(60, 100) / 365
        param_state = rmenv.DemandParameterState(any_arrival_rate, price_sensitivity=.4)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(any_arrival_rate, any_arrival_rate),
                                                                               (0.1, 0.9)])
        s_dummy, r_dummy, done_dummy, _ = dummy_env.step(0)

        s, r, done, _ = env.step(0)

        assert s[-1] == 2 * (.4 - .1) / (.9 - .1) - 1.
        assert s[0:-1] == (s_dummy,)
        assert r == r_dummy
        assert done == done_dummy

    def test_step_observations_are_wrapped_with_normalized_arrival_rate(self) -> None:
        any_price_sensitivity = random.random()
        param_state = rmenv.DemandParameterState(arrival_rate=1.3, price_sensitivity=any_price_sensitivity)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(
            dummy_env, param_state, params_range=[(1.1, 1.5), (any_price_sensitivity, any_price_sensitivity)])
        s_dummy, r_dummy, done_dummy, _ = dummy_env.step(0)

        s, r, done, _ = env.step(0)

        assert s[-1] == 2 * (1.3 - 1.1) / (1.5 - 1.1) - 1.
        assert s[0:-1] == (s_dummy,)
        assert r == r_dummy
        assert done == done_dummy

    def test_step_observations_are_wrapped_with_normalized_arrival_rate_and_price_sensitivity(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=1.3, price_sensitivity=0.4)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(
            dummy_env, param_state, params_range=[(1.1, 1.5), (0.1, 0.9)])
        s_dummy, r_dummy, done_dummy, _ = dummy_env.step(0)

        s, r, done, _ = env.step(0)

        assert (s[-1] == (2 * (1.3 - 1.1) / (1.5 - 1.1) - 1., 2 * (.4 - .1) / (.9 - .1) - 1.)).all()
        assert s[0:-1] == (s_dummy,)
        assert r == r_dummy
        assert done == done_dummy

    def test_reset_is_wrapped_with_price_sensitivity(self) -> None:
        any_arrival_rate = random.randint(60, 100) / 365
        param_state = rmenv.DemandParameterState(any_arrival_rate, price_sensitivity=.4)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(
            dummy_env, param_state, params_range=[(any_arrival_rate, any_arrival_rate), (0.1, 0.9)])
        state_dummy = dummy_env.reset()

        state = env.reset()

        assert state[-1] == 2 * (.4 - .1) / (.9 - .1) - 1.
        assert state[0:-1] == (state_dummy,)

    def test_reset_is_wrapped_with_arrival_rate(self) -> None:
        any_price_sensitivity = random.random()
        param_state = rmenv.DemandParameterState(arrival_rate=1.3, price_sensitivity=any_price_sensitivity)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(
            dummy_env, param_state, params_range=[(1.1, 1.5), (any_price_sensitivity, any_price_sensitivity)])
        state_dummy = dummy_env.reset()

        state = env.reset()

        assert state[-1] == 2 * (1.3 - 1.1) / (1.5 - 1.1) - 1.
        assert state[0:-1] == (state_dummy,)

    def test_reset_is_wrapped_with_arrival_rate_and_price_sensitivity(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=1.3, price_sensitivity=0.4)
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(
            dummy_env, param_state, params_range=[(1.1, 1.5), (0.1, 0.9)])
        state_dummy = dummy_env.reset()

        state = env.reset()

        assert (state[-1] == (2 * (1.3 - 1.1) / (1.5 - 1.1) - 1., 2 * (.4 - .1) / (.9 - .1) - 1.)).all()
        assert state[0:-1] == (state_dummy,)

    def test_observation_space_stacks_the_price_sensitivity(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(1.1, 1.1), (0.1, 0.9)])

        assert env.observation_space[0:-1] == (dummy_env.observation_space,)
        assert env.observation_space[-1] == spaces.Box(low=-1., high=1., shape=(1,), dtype=float)

    def test_observation_space_stacks_the_arrival_rate(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(1.1, 1.5), (0.1, 0.1)])

        assert env.observation_space[0:-1] == (dummy_env.observation_space,)
        assert env.observation_space[-1] == spaces.Box(low=-1., high=1., shape=(1,), dtype=float)

    def test_observation_space_stacks_the_arrival_rate_and_price_sensitivity(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(1.1, 1.5), (0.1, 0.9)])

        assert env.observation_space[0:-1] == (dummy_env.observation_space,)
        assert env.observation_space[-1] == spaces.Box(low=-1., high=1., shape=(2,), dtype=float)

    def test_observation_space_is_unchanged_if_no_params_can_change(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(1.1, 1.1), (0.1, 0.1)])

        assert env.observation_space == dummy_env.observation_space

    def test_action_space_is_unchanged(self) -> None:
        param_state = rmenv.DemandParameterState(arrival_rate=random.random(), price_sensitivity=random.random())
        dummy_env = tools.DummyEnv()
        env = rmenv.TrueParamsObsWrapper(dummy_env, param_state, params_range=[(1.1, 1.6), (0.1, 0.9)])

        assert env.action_space == dummy_env.action_space


if __name__ == '__main__':
    unittest.main()
