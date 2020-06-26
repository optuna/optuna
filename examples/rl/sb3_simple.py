""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


N_TRIALS = 100
N_JOBS = 2
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = "CartPole-v1"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_ID,
}


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters.

    :param trial: (optuna.Trial)
    :return: (Dict[str, Any])
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.999, 0.9999])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7, 0.9, 1, 2, 5])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    net_arch = (
        {"pi": [64], "vf": [64],} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64],}
    )

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU,}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune(self.eval_idx):
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters
    kwargs.update(sample_a2c_params(trial))
    # Create the RL model
    model = A2C(**kwargs)
    # Create env used for evaluation
    eval_env = gym.make(ENV_ID)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        # Prune hyperparams that generate NaNs
        print(e)
        raise optuna.exceptions.TrialPruned()
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
