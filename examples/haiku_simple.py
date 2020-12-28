"""
Optuna example that optimizes a neural network classifier configuration for the
MNIST dataset using Jax and Haiku.

In this example, we optimize the validation accuracy of MNIST classification using
jax nad haiku. We optimize the number of linear layers and learning rate of the optimizer.

The example code is based on https://github.com/deepmind/dm-haiku/blob/master/examples/mnist.py
"""

from typing import Any
from typing import Generator
from typing import Mapping
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

import optuna


OptState = Any
Batch = Mapping[str, np.ndarray]

BATCH_SIZE = 128
TRAIN_STEPS = 1000
N_TRAIN_SAMPLES = 3000
N_VALID_SAMPLES = 1000


def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    sample_size: int,
) -> Generator[Batch, None, None]:
    """Loads the sub-sampled dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).take(sample_size).cache().repeat()
    if is_training:
        ds = ds.shuffle(sample_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def objective(trial):
    # Make datasets.
    train = load_dataset(
        "train", is_training=True, batch_size=BATCH_SIZE, sample_size=N_TRAIN_SAMPLES
    )
    train_eval = load_dataset(
        "train", is_training=False, batch_size=BATCH_SIZE, sample_size=N_TRAIN_SAMPLES
    )
    test_eval = load_dataset(
        "test", is_training=False, batch_size=BATCH_SIZE, sample_size=N_VALID_SAMPLES
    )

    # Draw hyper-parameters
    n_units_l1 = trial.suggest_int("n_units_l1", 4, 128)
    n_units_l2 = trial.suggest_int("n_units_l2", 4, 128)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Define feed-forward function by using sampled parameters
    def net_fn(batch: Batch) -> jnp.ndarray:
        """Standard MLP network."""
        x = batch["image"].astype(jnp.float32) / 255.0
        mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(n_units_l1),
                jax.nn.relu,
                hk.Linear(n_units_l2),
                jax.nn.relu,
                hk.Linear(10),
            ]
        )
        return mlp(x)

    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(lr)

    # Training loss (cross-entropy).
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * l2_loss

    # Evaluation metric (classification accuracy).
    @jax.jit
    def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: OptState,
        batch: Batch,
    ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # Initialize network and optimiser; note we draw an input to get shapes.
    params = net.init(jax.random.PRNGKey(42), next(train))
    opt_state = opt.init(params)

    best_test_accuracy = 0.0
    # Train/eval loop.
    for step in range(1, TRAIN_STEPS + 1):
        if step % 100 == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_accuracy = accuracy(params, next(train_eval))
            test_accuracy = accuracy(params, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))

            print(
                f"[Step {step:5d}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}."
            )

            # Handle pruning based on the intermediate value.
            trial.report(test_accuracy, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_test_accuracy = max(best_test_accuracy, test_accuracy)

        # Do SGD on a batch of training examples.
        params, opt_state = update(params, opt_state, next(train))

    return best_test_accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, interval_steps=1000),
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
