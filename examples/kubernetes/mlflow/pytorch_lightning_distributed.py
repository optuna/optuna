import os

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna
from optuna.integration.mlflow import MLflowCallback


PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 5
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class MetricsCallback(Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    input_dim = 28 * 28
    layers = []

    for i in range(n_layers):
        output_dim = int(trial.suggest_float("n_units_l{}".format(i), 4, 128))
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        input_dim = output_dim

    layers.append(nn.Linear(input_dim, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)

    return model


class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super().__init__()
        self.model = create_model(trial)

    def forward(self, data):
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        return {"loss": F.nll_loss(output, target)}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean().item()
        return {"batch_val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x["batch_val_acc"] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {"log": {"val_acc": accuracy}}

    def configure_optimizers(self):
        return Adam(self.model.parameters())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=False,
        )


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        val_percent_check=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=None,
        callbacks=[metrics_callback],
    )

    model = LightningNet(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_acc"]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="k8s_mlflow",
        storage="postgresql://{}:{}@postgres:5432/{}".format(
            os.environ["POSTGRES_USER"],
            os.environ["POSTGRES_PASSWORD"],
            os.environ["POSTGRES_DB"],
        ),
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(
        objective,
        n_trials=100,
        timeout=600,
        callbacks=[MLflowCallback(
            tracking_uri="http://mlflow:5000/",
            metric_name="val_accuracy",
        )]
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
