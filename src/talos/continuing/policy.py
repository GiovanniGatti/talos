import abc
from typing import Union, List, Dict, Optional, Tuple, Callable, Any

import gym
import numpy as np
from ray import rllib
from ray.rllib import SampleBatch, evaluation
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, ModelWeights, ModelGradients
from scipy import stats, optimize

from talos import dp
from talos.continuing import policy_state


class _NotLearnablePolicy(rllib.Policy, abc.ABC):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)

    @abc.abstractmethod
    def actions(self, obs_batch: Union[List[TensorType], TensorType]) -> np.ndarray:
        pass

    def compute_log_likelihoods(self,
                                actions: Union[List[TensorType], TensorType],
                                obs_batch: Union[List[TensorType], TensorType],
                                state_batches: Optional[List[TensorType]] = None,
                                prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
                                prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
                                actions_normalized: bool = True) -> TensorType:
        pass

    def compute_actions(self,
                        obs_batch: Union[List[TensorType], TensorType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorType], TensorType] = None,
                        prev_reward_batch: Union[List[TensorType], TensorType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List[evaluation.MultiAgentEpisode]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        actions = self.actions(obs_batch)
        return actions, [], {}

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        pass

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        pass

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass

    def compute_gradients(self, postprocessed_batch: SampleBatch) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        return [], {}

    def apply_gradients(self, gradients: ModelGradients) -> None:
        pass

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        return {}

    def get_weights(self) -> ModelWeights:
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        pass

    def export_model(self, export_dir: str, onnx: Optional[int] = None) -> None:
        pass

    def export_checkpoint(self, export_dir: str) -> None:
        pass

    def import_model_from_h5(self, import_file: str) -> None:
        pass


class RmsPolicy(_NotLearnablePolicy):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        env_config = config['env_config']

        assert not env_config['one-hot'], 'This policy implementation works only with min-max normalized spaces'
        assert not env_config['with-true-params'], 'Use OptimalPolicy instead!'
        assert 'forecasting' in env_config, \
            'This implementation works only within strategic learning envs or with forecasting'

        self._horizon = env_config['booking-horizon']
        self._initial_capacity = env_config['initial-capacity']
        self._fare_structure = np.array(env_config['fare-structure'], dtype=float)
        self._arrival_rate = [arr / self._horizon for arr in env_config['mean-arrivals']]

        self._state_max = np.array([self._horizon - 1, self._initial_capacity])

        _config = env_config['forecasting']
        params_range = _config['params-range']
        self._n_params = len(params_range)

        self._params_range = np.array(params_range)
        self._max = np.max(self._params_range, axis=1)
        self._min = np.min(self._params_range, axis=1)

    def actions(self, obs_batch: Union[List[TensorType], TensorType]) -> np.ndarray:
        batch_size = obs_batch.shape[0]

        states = obs_batch[:, :2 * self._horizon].reshape(batch_size, self._horizon, 2)
        states = np.rint((states + 1) * self._state_max / 2).astype(int)

        params = obs_batch[:, 2 * self._horizon:2 * self._horizon + self._n_params].reshape(batch_size, self._n_params)
        params = (params + 1) / 2 * (self._max - self._min) + self._min

        actions = []
        for s, _params in zip(states, params):
            lam, ps = _params if self._n_params > 1 else (self._arrival_rate[0], _params)
            q = dp.time_constant_dynamic_programming(
                self._initial_capacity, lam, ps, self._fare_structure, self._horizon)
            pi = q.argmax(-1)
            actions.append(pi[s[:, 0], s[:, 1]])
        return np.array(actions, dtype=int)


class OptimalPolicy(_NotLearnablePolicy):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        env_config = config['env_config']

        assert not env_config['one-hot'], 'This policy implementation works only with min-max normalized spaces'
        assert env_config['with-true-params'], 'Use RmsPolicy instead!'

        self._horizon = env_config['booking-horizon']
        self._initial_capacity = env_config['initial-capacity']
        self._fare_structure = tuple(env_config['fare-structure'])
        self._price_sensitivity = env_config['price-sensitivity']
        self._arrival_rate = [arr / self._horizon for arr in env_config['mean-arrivals']]

        self._state_max = np.array([self._horizon - 1, self._initial_capacity])

        self._arrival_rate_range = np.array([min(self._arrival_rate), max(self._arrival_rate)])
        self._price_sensitivity_range = np.array([min(self._price_sensitivity), max(self._price_sensitivity)])

        if len(self._arrival_rate) == 1 and len(self._price_sensitivity) == 1:
            _arrival_rate = self._arrival_rate[0]
            _price_sensitivity = self._price_sensitivity[0]

            def denormalize_fn(obs_batch: np.ndarray) -> np.ndarray:
                y = np.empty((obs_batch.shape[0], 2))
                y[:, 0] = _arrival_rate
                y[:, 1] = _price_sensitivity
                return y
        elif len(self._arrival_rate) == 1:
            _arrival_rate = self._arrival_rate[0]

            def denormalize_fn(obs_batch: np.ndarray) -> np.ndarray:
                batch_size = obs_batch.shape[0]
                y = np.empty((batch_size, 2))
                y[:, 0] = _arrival_rate
                price_sensitivity = obs_batch[:, 2 * self._horizon].reshape(batch_size, 1)
                price_sensitivity = (price_sensitivity + 1) / 2 * (
                        self._price_sensitivity_range[1] - self._price_sensitivity_range[0]) + \
                                    self._price_sensitivity_range[0]
                y[:, 1] = price_sensitivity[:, 0]
                return y
        elif len(self._price_sensitivity) == 1:
            _price_sensitivity = self._price_sensitivity[0]

            def denormalize_fn(obs_batch: np.ndarray) -> np.ndarray:
                batch_size = obs_batch.shape[0]
                y = np.empty((batch_size, 2))
                arrival_rate = obs_batch[:, 2 * self._horizon].reshape(batch_size, 1)
                arrival_rate = (arrival_rate + 1) / 2 * (
                        self._arrival_rate_range[1] - self._arrival_rate_range[0]) + self._arrival_rate_range[0]
                y[:, 0] = arrival_rate[:, 0]
                y[:, 1] = _price_sensitivity
                return y
        else:
            def denormalize_fn(obs_batch: np.ndarray) -> np.ndarray:
                batch_size = obs_batch.shape[0]
                y = np.empty((batch_size, 2))
                arrival_rate = obs_batch[:, 2 * self._horizon].reshape(batch_size, 1)
                arrival_rate = (arrival_rate + 1) / 2 * (
                        self._arrival_rate_range[1] - self._arrival_rate_range[0]) + self._arrival_rate_range[0]
                y[:, 0] = arrival_rate[:, 0]
                price_sensitivity = obs_batch[:, 2 * self._horizon + 1].reshape(batch_size, 1)
                price_sensitivity = (price_sensitivity + 1) / 2 * (
                        self._price_sensitivity_range[1] - self._price_sensitivity_range[0]) + \
                                    self._price_sensitivity_range[0]
                y[:, 1] = price_sensitivity[:, 0]
                return y

        self._denormalize_fn = denormalize_fn

    def actions(self, obs_batch: Union[List[TensorType], TensorType]) -> np.ndarray:
        batch_size = obs_batch.shape[0]

        states = obs_batch[:, :2 * self._horizon].reshape(batch_size, self._horizon, 2)
        states = np.rint((states + 1) * self._state_max / 2).astype(int)

        params = self._denormalize_fn(obs_batch)

        actions = []
        for s, p in zip(states, params):
            pi, _ = policy_state.dynamic_programming(
                self._horizon, self._initial_capacity, p[0].item(), p[1].item(), self._fare_structure)
            actions.append(pi[s[:, 0] + 1, s[:, 1]])
        return np.array(actions, dtype=int)


class MaximizeRevenueWithStdPenaltyPolicy(_NotLearnablePolicy):
    """
    Implementation of form2 from 'Novel pricing strategies for revenue maximization and demand
    learning using an exploration–exploitation framework'
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        env_config = config['env_config']

        assert not env_config['one-hot'], 'This policy implementation works only with min-max normalized spaces'
        assert not env_config['with-true-params'], 'Use OptimalPolicy instead!'

        assert not env_config['one-hot'], 'This policy implementation works only with min-max normalized spaces'

        assert len(env_config['mean-arrivals']) == 1, \
            'The Average Policy implementation supports only uncertainty over the price sensitivity'

        assert 'forecasting' in env_config, 'This implementation works only with forecasting'

        forecasting = env_config['forecasting']

        assert 'eta' in env_config, 'Expected the penalty exploration parameter (eta) to be configured'

        self._horizon = env_config['booking-horizon']
        self._initial_capacity = env_config['initial-capacity']
        self._fare_structure = np.array(env_config['fare-structure'], dtype=np.float64)
        self._f0 = np.min(self._fare_structure)
        self._arrival_rate = env_config['mean-arrivals'][0] / self._horizon
        self._eta = env_config['eta']

        params_range = forecasting['params-range'][0]
        self._min_ps, self._max_ps = min(params_range), max(params_range)

        assert stats.poisson.cdf(mu=env_config['mean-arrivals'][0], k=self._initial_capacity) > 0.995, \
            'This method supports only unconstrained capacity'

        self._action_indexes = np.arange(self._fare_structure.shape[0])

    def actions(self, obs_batch: Union[List[TensorType], TensorType]) -> np.ndarray:
        batch_size = obs_batch.shape[0]

        price_sensitivity = obs_batch[:, 2 * self._horizon:2 * self._horizon + 1].reshape(batch_size, 1)
        price_sensitivity = (price_sensitivity + 1) / 2 * (self._max_ps - self._min_ps) + self._min_ps

        offers = obs_batch[:, 2 * self._horizon + 1:].reshape(batch_size, self._fare_structure.shape[0])

        actions = []
        for ps, o in zip(price_sensitivity, offers):
            action = self._std_optimization(ps.item(), o)
            actions.append(action)

        return np.array(actions, dtype=int)

    def _std_optimization(self, price_sensitivity: float, offers: np.ndarray) -> np.ndarray:
        d = self._arrival_rate * np.exp(-price_sensitivity * (self._fare_structure / self._f0 - 1))
        exp_rev = self._fare_structure * d

        def fn(x: np.ndarray) -> float:
            information = np.sum((self._fare_structure / self._f0 - 1) ** 2 * (offers + x) * d)
            information = np.clip(information, a_min=0.001, a_max=None)
            std = 1. / np.sqrt(information)
            cov = std / price_sensitivity
            rev = np.sum(x * exp_rev)
            return - rev + self._eta * cov

        def jac(x: np.ndarray) -> np.ndarray:
            information = np.sum((self._fare_structure / self._f0 - 1) ** 2 * (offers + x) * d)
            information = np.clip(information, a_min=0.001, a_max=None)
            std = 1. / np.sqrt(information)
            dx = 1. / (2 * price_sensitivity) * std ** 3 * (self._fare_structure / self._f0 - 1) ** 2 * d
            return - exp_rev - self._eta * dx

        return self._optimize(fn, jac)

    def _optimize(self, objective_fn: Callable[[Any], float], jac: Callable[[Any], np.ndarray]) -> np.ndarray:
        x0 = np.random.uniform(size=self._fare_structure.shape[0])
        x0 = self._horizon * x0 / np.sum(x0)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self._horizon, 'jac': lambda x: np.ones_like(x)},)
        constraints += ({'type': 'ineq', 'fun': lambda x: x, 'jac': lambda x: np.eye(x.shape[0])},)
        # noinspection PyTypeChecker
        # following scipy doc
        res = optimize.minimize(objective_fn, x0, method='SLSQP', constraints=constraints, jac=jac, tol=1e-12)

        weights = res.x
        weights = np.clip(weights / self._horizon, a_min=0., a_max=1.)
        weights = weights / np.sum(weights)

        actions = np.random.choice(self._action_indexes, size=self._horizon, p=weights)

        return actions
