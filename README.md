# Talos
This repository contains the code used in the scientific publication "[Outsmarting human design in airline revenue management](https://www.mdpi.com/1999-4893/15/5/142)". In this paper, we trained an artificial neural network (ANN) to tackle the earning-while-learning problem in airline revenue management (RM) through reinforcement learning (RL).
Roughly, the idea behind the paper is to demonstrate that RL can be used to find novel solutions to airline revenue management problems that can outperform human-designed heuristic-based solutions.

According to our proposal, the ANN must be trained before deployment in the real world. For this reason, this repository contains two execution modes. The first, `continuing`, refers to the training of the ANN, while the second, `rollout`, evaluates the various methods (RL, classical RM solutions, and heuristic methods).

Because not everyone has access to the kind of computational power needed to reproduce our work, we are also making available the training checkpoints for each scenario in the paper in the folder `training-checkpoints`. You can use it to evaluate the agent's ability to handle the earning while learning problems at any saved checkpoint during the training procedure. To reproduce the images from the paper, we advise using the last available checkpoint.

The code has been develop with [RLLib](https://www.ray.io/rllib). Please refer to their documentation for details.

# Scripts
This section presents how to use the training/evaluation scripts.

## `entry/continuing.py`
This script is to be used to train the ANN. It is up to the user to set up the training environment parameters. We advise following the described in the paper. There are 

 + eight arguments (`--batch-size`, `--minibatch-size`, `--lr`, `--gamma`, etc), related to RL hyperparameters settings;
 + two arguments (`--model`, `--lstm-cell-size`) that defines the artificial neural network topology;
 + four arguments that define the environment settings (`--initial-capacity`, `--horizon`, etc);
 + one argument for selecting the reward signal `--rwd-fn`;
 + a couple of miscellaneous arguments on post-processing of rewards (i.e., average reward), how long the training must continue, etc.
 
Use the option `--help` for more details.

## `entry/rollout.py`
This script evaluates the various methods presented in the paper. There are essentially four policies of interest

1. optimal policy (must set `--with-true-params`): is the policy that knows the true demand behavior parameters at all times;
2. RMS policy (must set `--with-forecasting`, `--mean-arrivals-range` and `--frat5-range`): follows the classical RMS loop, which forecasts demand and optimizes for it through dynamic programming;
3. heuristic method (must set `--with-forecasting`, `--mean-arrivals`, `--frat5-range` and `--eta`): optimizes the policy that balances between model uncertainty and revenue maximization. Recall that the heuristic method is only available if the expected number of customer arrivals is much smaller than the available capacity;
4. RL (must set `--checkpoint` and `--params-file`): load the trained policy from the provided checkpoint.

For example, to load one of the checkpoints and evaluate the RL agent, one can do so with

```bash
python entry/rollout.py --mean-arrivals 4 --with-forecasting --frat5-range 1.5 4.3 --checkpoint training-checkpoints/single-parameter-estimation/PPO_with-uniform-sampling_d5114_00000_0_2021-12-20_08-21-29/checkpoint_000715/checkpoint-715 --params-file training-checkpoints/single-parameter-estimation/PPO_with-uniform-sampling_d5114_00000_0_2021-12-20_08-21-29/params.json --output-dir /tmp/
```

For the other arguments, please refer to `--help` for details.

This script produces a set of graphs and metrics useful for general analysis. It also computes default metrics, such as the average obtained revenue, load factor, etc.

# Setting up the virtual environment
If you plan to do some development, we recommend installing the dependencies in the `requirements.txt` file in a [python virtual environment](https://docs.python.org/3/tutorial/venv.html), i.e,

```bash
pip install -r requirements.txt
pip install -e .
```

If your goal is only to run the code, you can simply install it directly from the `setup.py`, i.e.
```bash
pip install setup.py
```

Please, be sure to have TensorFlow installed with a [compatible version](https://www.tensorflow.org/install/).

# Running unit tests
We developed some unit tests for the most critical parts of the code. You can run them with

```bash
pytest -n <num-of-cores-to-distribute-computation>
```

Some of these critical functions carry some stochastic behavior (e.g., demand generation), and thus we perform statistical tests. In other words, we compare the average behavior of these functions with the expected theoretical results. There are two downsides to such an approach:
To collect data needed for the tests, we need to call these functions many times, and this comes with computational costs;
Tests may be unreliable, failing once in a while due to false positives. The workarounds to this issue are to either relax confidence levels of the tests (which degrades the confidence in the tests themselves) or increase sample sizes, which come with more computational costs.
We tuned confidence levels and sample sizes according to the computational resources available to us. You might find a different trade-off to be better according to the hardware you have.