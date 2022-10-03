import argparse
import gzip
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import random
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, Optional, Type, Callable

import numpy as np
import ray
from matplotlib import pyplot as plt
from ray import rllib
from ray.rllib import policy, Policy
from ray.rllib.agents import Trainer, ppo, trainer
from ray.rllib.agents.ppo import PPOTFPolicy
from ray.rllib.evaluation import worker_set
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import sample_batch
from ray.rllib.policy import view_requirement
from statsmodels.stats import weightstats

from talos.continuing import model, env
from talos.continuing.policy import RmsPolicy, OptimalPolicy, MaximizeRevenueWithStdPenaltyPolicy

CheckpointLoader = Callable[[rllib.Policy], None]


def isfile(path: str) -> str:
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'file path:{path} is not a valid file')


def get_params(params_file: str) -> Dict[Any, Any]:
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def _update_requirements(agent_policy: policy.Policy, policy_id: sample_batch.PolicyID) -> None:
    agent_policy.view_requirements[sample_batch.SampleBatch.INFOS] = \
        view_requirement.ViewRequirement(sample_batch.SampleBatch.INFOS, shift=0)


def rollout(_policy_class: Optional[Type[Policy]],
            _env_creator: Callable[[rllib.env.EnvContext], Any],
            _trainer_config: Dict[str, Any],
            _checkpoint_loader: Optional[CheckpointLoader] = None) -> sample_batch.SampleBatch:
    if not ray.is_initialized():
        raise RuntimeError('Initialize ray first!')

    num_cpus = multiprocessing.cpu_count()
    num_workers = num_cpus - 1

    ModelCatalog.register_custom_model('time-distributed', model.TimeDistributedModel)
    ModelCatalog.register_custom_model('encoder-decoder', model.EncoderDecoderModel)

    workers = worker_set.WorkerSet(policy_class=_policy_class,
                                   env_creator=_env_creator,
                                   num_workers=num_workers,
                                   logdir='./logs/ray-results/',
                                   trainer_config=_trainer_config)

    policy = workers.local_worker().get_policy()
    if _checkpoint_loader:
        _checkpoint_loader(policy)
        workers.sync_weights()

    workers.foreach_policy(_update_requirements)

    return sample_batch.SampleBatch.concat_samples(ray.get([w.sample.remote() for w in workers.remote_workers()]))


def export_metrics(batch: sample_batch.SampleBatch, output_dir: str) -> None:
    metrics = defaultdict(list)
    for episode in batch.split_by_episode():
        for k in {'orig_obs', 'optimal_action', 'raw_rev', 'rwd_before_norm', 'sampled_price_sensitivity',
                  'price_sensitivity_std', 'incomplete_flights', 'estimated_price_sensitivity',
                  'price_sensitivity_error', 'estimated_arrival_rate', 'arrival_rate_std', 'arrival_rate_error',
                  'accuracy'}:
            if k not in episode['infos'][0]:
                continue
            metrics[k].append([info[k] for info in episode['infos']])
        metrics['actions'].append(episode['actions'])

    for k in ['rwd_before_norm', 'optimal_action', 'price_sensitivity_std', 'raw_rev', 'estimated_price_sensitivity',
              'price_sensitivity_error', 'estimated_arrival_rate', 'arrival_rate_std', 'arrival_rate_error',
              'accuracy']:
        if k not in metrics:
            continue
        m = np.squeeze(np.array(metrics[k])[:, 3 * booking_horizon:])
        m += np.random.normal(loc=0, scale=1 / np.sqrt(m.shape[0]), size=m.shape)
        stats = weightstats.DescrStatsW(np.mean(m, axis=-1))
        mean = stats.mean
        l, u = stats.tconfint_mean(alpha=.01)
        median = np.median(m)

        fig, ax = plt.subplots()
        ax.hist(m.flatten(), bins=100, density=True)
        ax.set_ylabel('Probability')
        ax.set_xlabel(k)
        fig.savefig(str(pathlib.Path(output_dir, '{}_hist.jpg'.format(k))))

        print('{}: {} \u00b1 {}, var={}, median={}'.format(k, mean, (u - l) / 2, stats.var, median))

    remaining_cap = np.array([[t[0][0][1] for t in ep] for ep in metrics['orig_obs']])
    remaining_cap = remaining_cap[:, 3 * booking_horizon:].flatten()
    stats = weightstats.DescrStatsW(1. - remaining_cap / initial_capacity)
    mean = stats.mean
    l, u = stats.tconfint_mean(alpha=.01)
    print('load_factor: {} \u00b1 {}'.format(mean, (u - l) / 2))

    if not namespace.skip_data_export:
        with gzip.open(str(pathlib.Path(output_dir, 'evaluation.pickle.gz')), 'wb') as zipfile:
            pickle.dump(metrics, zipfile)


def export_fare_class_distribution(
        env_config: Dict[str, Any], batch: sample_batch.SampleBatch, output_dir: str
) -> None:
    fare_structure = env_config['fare-structure']
    size = len(fare_structure)

    incomplete_flights = np.array([info['incomplete_flights'] for info in batch['infos']]).flatten()

    actions = batch['actions'].flatten()
    actions = actions[incomplete_flights]
    bins, counts = np.unique(actions, return_counts=True)
    counts = counts / np.sum(counts)
    hist = np.zeros(size)
    hist[bins] = counts

    optimal_policy = np.array([info['opt_policy'] for info in batch['infos']]).flatten()
    optimal_policy = optimal_policy[incomplete_flights]
    bins, counts = np.unique(optimal_policy, return_counts=True)
    counts = counts / np.sum(counts)
    opt_hist = np.zeros(size)
    opt_hist[bins] = counts

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(5, 15))
    for ax in axs:
        ax.grid(b=True, which='major', color='#445577', linestyle=':')
    width = 1.
    x = np.arange(len(fare_structure))

    axs[0].bar(x, opt_hist, width, alpha=.5, color='green')
    axs[1].bar(x, hist, width, alpha=.5, color='blue')

    axs[2].bar(x, opt_hist, width, alpha=.7, color='green')
    axs[2].bar(x, hist, width, alpha=.3, color='blue')

    axs[0].set_ylim((0., 1.))
    axs[1].set_ylim((0., 1.))
    axs[2].set_ylim((0., 1.))

    axs[0].set_yticks(np.arange(0, 1, .1))
    axs[1].set_yticks(np.arange(0, 1, .1))
    axs[2].set_yticks(np.arange(0, 1, .1))

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(fare_structure)

    fig.savefig(str(pathlib.Path(output_dir, 'hist.jpg')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num-episodes', type=int, default=3_000, help='Number of episodes to simulate')
    parser.add_argument('--skip-data-export', action='store_true', help='Skip the generation of export data')

    parser.add_argument('--initial-capacity', type=int, default=50, help='The leg capacity')
    parser.add_argument('--horizon', type=int, default=22, help='The booking horizon')
    parser.add_argument('--mean-arrivals', type=float, default=[55., 80.], nargs='+',
                        help='If using the discrete environment, it\'s a list of mean-arrival to train the agent with. '
                             'If using the uniform environment, it\'s the min/max range for mean arrivals')
    parser.add_argument('--frat5', type=float, nargs='+', default=[2.1, 3.8],
                        help='If using the discrete environment, it\'s a list of frat5s to train the agent with. '
                             'If using the uniform environment, it\'s the min/max range for the frat5')

    parser.add_argument('--one-hot', action='store_true', help='Activate one-hot encoding for agent observations')

    parser.add_argument('--env', type=str, default='with-uniform-sampling',
                        choices=['with-discrete-sampling', 'with-uniform-sampling'],
                        help='The training environment setting to use')

    parser.add_argument('--warmup-policy', type=str, choices=['optimal', 'random'],
                        help='Policy to be used when initializing the environment. If unspecified, '
                             'the environment will use a soft-initialization, i.e., there will be no refresh on '
                             'historical data, each new training episode continues from the previous one.')
    parser.add_argument('--with-true-params', action='store_true', help='Use the optimal policy')

    parser.add_argument('--debugging', action='store_true', help='Run locally with simplified settings')

    parser.add_argument('--checkpoint', type=file_path, help='The path to the checkpoint file')
    parser.add_argument('--params-file', type=file_path, help='The path to the original training parameters file '
                                                              '(generated by RLLib when training)')

    parser.add_argument('--output-dir', type=dir_path, required=True, help='Where to export charts')

    # forecasting
    parser.add_argument('--with-forecasting', action='store_true',
                        help='Use price sensitivity estimation module from RMS')
    parser.add_argument('--mean-arrivals-range', type=float, nargs=2,
                        help='The range of true mean-arrivals (applicable when simulating RMS in forecasting mode '
                             'and single mean-arrivals value)')
    parser.add_argument('--frat5-range', type=float, nargs=2,
                        help='The range of true frat5s (applicable when simulating RMS in forecasting mode '
                             'and single frat5 value)')

    # maximize revenue with std penalty
    parser.add_argument('--eta', type=float,
                        help='Activate Policy that maximizes revenue while controlling std error through a penalty')

    namespace = parser.parse_args(sys.argv[1:])
    num_episodes = namespace.num_episodes
    num_cpus = multiprocessing.cpu_count()
    num_workers = min(num_cpus - 1, num_episodes)

    env_config = {
        'booking-horizon': namespace.horizon,
        'initial-capacity': namespace.initial_capacity,
        'mean-arrivals': namespace.mean_arrivals,
        'rwd-fn': 'expected-rwd',
        'price-sensitivity': [math.log(2) / (frat5 - 1.) for frat5 in namespace.frat5],
        'one-hot': namespace.one_hot,
        'discount-rate': random.random(),  # this value is unimportant for policy rollouts
        'fare-structure': tuple(range(50, 250, 20)),
        'with-true-params': namespace.with_true_params,
        'warmup-policy': namespace.warmup_policy,
    }

    initial_capacity: int = namespace.initial_capacity
    booking_horizon: int = namespace.horizon

    ray.init(local_mode=namespace.debugging)

    if namespace.checkpoint:
        assert namespace.params_file, 'The parameters file must be specified when loading checkpoints.'
        policy_class = PPOTFPolicy
        _path = namespace.checkpoint

        original_config = get_params(namespace.params_file)
        original_env_config = original_config['env_config']

        if 'forecasting' in original_env_config:
            env_config['forecasting'] = original_env_config['forecasting']

        trainer_config = Trainer.merge_trainer_configs(
            ppo.DEFAULT_CONFIG, {
                'num_envs_per_worker': int(num_episodes / num_workers),
                'rollout_fragment_length': 20 * booking_horizon,
                'horizon': 20 * booking_horizon,
                'soft_horizon': False,
                'no_done_at_end': True,
                'framework': 'tf',
                'model': original_config['model'],
                'env_config': env_config})


        def checkpoint_loader(_policy: PPOTFPolicy) -> None:
            extra_data = pickle.load(open(_path, 'rb'))
            worker = pickle.loads(extra_data['worker'])
            _policy.set_state(worker['state']['default_policy'])
    else:
        if namespace.with_forecasting:
            assert namespace.frat5_range is not None, 'RMS + forecasting require specifying frat5 range'

            params_range = []
            if namespace.mean_arrivals_range is not None:
                assert min(namespace.mean_arrivals_range) <= min(namespace.mean_arrivals)
                assert max(namespace.mean_arrivals_range) >= max(namespace.mean_arrivals)

                arrival_rate_range = min(namespace.mean_arrivals_range) / booking_horizon, \
                                     max(namespace.mean_arrivals_range) / booking_horizon
                params_range += [arrival_rate_range, ]

            assert min(namespace.frat5_range) <= min(namespace.frat5)
            assert max(namespace.frat5_range) >= max(namespace.frat5)

            price_sensitivity_range = math.log(2) / (max(namespace.frat5_range) - 1), \
                                      math.log(2) / (min(namespace.frat5_range) - 1)
            params_range += [price_sensitivity_range, ]

            env_config['forecasting'] = {
                'params-range': params_range,
            }

        policy_class = OptimalPolicy if namespace.with_true_params else \
            MaximizeRevenueWithStdPenaltyPolicy if namespace.eta is not None else RmsPolicy

        if namespace.eta is not None:
            env_config['eta'] = namespace.eta

        trainer_config = Trainer.merge_trainer_configs(
            trainer.COMMON_CONFIG, {
                'num_envs_per_worker': int(num_episodes / num_workers),
                'rollout_fragment_length': 20 * booking_horizon,
                'horizon': 20 * booking_horizon,
                'soft_horizon': False,
                'no_done_at_end': True,
                'framework': 'tf2',
                'env_config': env_config})

        checkpoint_loader = None

    output_dir = tempfile.mkdtemp() if namespace.debugging else namespace.output_dir

    if namespace.env == 'with-discrete-sampling':
        env_creator = env.with_discrete_sampling
    elif namespace.env == 'with-uniform-sampling':
        env_creator = env.with_uniform_sampling
    else:
        raise ValueError('Unexpected environment {}'.format(namespace.env))

    batch = rollout(policy_class, env_creator, trainer_config, checkpoint_loader)

    export_fare_class_distribution(env_config, batch, output_dir)
    export_metrics(batch, output_dir)
