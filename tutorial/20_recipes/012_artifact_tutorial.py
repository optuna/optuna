"""
.. _artifact_tutorial:

Optuna Artifacts Tutorial
=========================

.. contents:: Table of Contents
    :depth: 2

The artifact module of Optuna is a module designed for saving comparatively large attributes on a trial-by-trial basis in forms such 
as files. Introduced from Optuna v3.3, this module finds a broad range of applications, such as utilizing snapshots of large size 
models for hyperparameter tuning, optimizing massive chemical structures, and even human-in-the-loop optimization employing images 
or sounds. Use of Optuna's artifact module allows you to handle data that would be too large to store in a database. Furthermore, 
by integrating with `optuna-dashboard <https://github.com/optuna/optuna-dashboard>`_, saved artifacts can be automatically visualized 
with the web UI, which significantly reduces the effort of experiment management.

TL;DR
-----

- The artifact module provides a simple way to save and use large data associated with trials.

- Saved artifacts can be visualized just by accessing the web page using optuna-dashboard, and downloading is also easy.

- Thanks to the abstraction of the artifact module, the backend (file system, AWS S3) can be easily switched.

- As the artifact module is tightly linked with Optuna, experiment management can be completed with the Optuna ecosystem alone, simplifying the code base.

Concepts
--------

.. list-table::
    :header-rows: 1

    * - Fig 1. Concepts of the "artifact".
    * - .. image:: https://github.com/optuna/optuna/assets/38826298/112e0b75-9d22-474b-85ea-9f3e0d75fa8d

An "artifact" is associated with an Optuna trial. In Optuna, the objective function is evaluated sequentially to search for the 
maximum (or minimum) value. Each evaluation of the sequentially repeated objective function is called a trial. Normally, trials and 
their associated attributes are saved via storage objects to files or RDBs, etc. For experiment management, you can also save and 
use `user_attrs` for each trial. However, these attributes are assumed to be integers, short strings, or other small data, which 
are not suitable for storing large data. With Optuna's artifact module, users can save large data (such as model snapshots, 
chemical structures, image and audio data, etc.) for each trial.

Also, while this tutorial does not touch upon it, it's possible to manage artifacts associated not only with trials but also with 
studies. Please refer to the `official documentation <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.artifacts.upload_artifact.html>`_ 
if you are interested in.

Situations where artifacts are useful
-------------------------------------

Artifacts are useful when you want to save data that is too large to be stored in RDB for each trial. For example, the artifact 
module would be handy in situations like the following:

- Saving snapshots of machine learning models: Suppose you are tuning hyperparameters for a large-scale machine learning model like 
  an LLM. The model is very large, and each round of learning (which corresponds to one trial in Optuna) takes time. To prepare for 
  unexpected incidents during training (such as blackouts at the data center or a preemption of computation jobs by the scheduler), 
  you may want to save snapshots of the model in the middle of training for each trial. These snapshots often tend to be large and 
  are more suitable to be saved as some kinds of files than to be stored in RDB. In such cases, the artifact module is useful.

- Optimizing chemical structures: Suppose you are formulating and exploring a problem of finding stable chemical structures as a 
  black-box optimization problem. Evaluating one chemical structure corresponds to one trial in Optuna, and that chemical structure 
  is a complex and large one. It is not appropriate to store such chemical structure data in RDB. It is conceivable to save the 
  chemical structure data in a specific file format, and in such a case, the artifact module is useful.

- Human-in-the-loop optimization of images: Suppose you are optimizing prompts for a generative model that outputs images. You 
  sample the prompts using Optuna, output images using the generative model, and let humans rate the images for a Human-in-the-loop 
  optimization process. Since the output images are large data, it is not appropriate to use RDB to store them, and in such cases, 
  using the artifact module is well suited.

How Trials and Artifacts are Recorded
-------------------------------------

As explained so far, the artifact module is useful when you want to save large data for each trial. In this section, we explain
how artifacts work in the following two scenarios: first when SQLite + local file system-based artifact backend is used
(suitable when the entire optimization cycle is completed locally), and second when MySQL + AWS S3-based artifact backend is used
(suitable when you want to keep the data in a remote location).

Scenario 1: SQLite + file system-based artifact store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Fig 2. SQLite + file system-based artifact store.
    * - .. image:: https://github.com/optuna/optuna/assets/38826298/d41d042e-6b78-4615-bf96-05f73a47e9ea

First, we explain a simple case where the optimization is completed locally.

Normally, Optuna's optimization history is persisted into some kind of a database via storage objects. Here, let's consider a 
method using SQLite, a lightweight RDB management system, as the backend. With SQLite, data is stored in a single file (e.g., 
./example.db). The optimization history comprises what parameters were sampled in each trial, what the evaluation values for those 
parameters were, when each trial started and ended, etc. This file is in the SQLite format, and it is not suitable for storing 
large data. Writing large data entries may cause performance degradation. Note that SQLite is not suitable for distributed parallel 
optimization. If you want to perform that, please use MySQL as we will explain later, or JournalStorage 
(`example <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html#optuna.storages.JournalStorage>`_).

So, let's use the artifact module to save large data in a different format. Suppose the data is generated for each trial and you 
want to save it in some format (e.g., png format if it's an image). The specific destination for saving the artifacts can be any 
directory on the local file system (e.g., the ./artifacts directory). When defining the objective function, you only need to save 
and reference the data using the artifact module.

The simple pseudocode for the above case  would look something like this:

.. code-block:: python

    import os

    import optuna
    from optuna.artifacts import FileSystemArtifactStore
    from optuna.artifacts import upload_artifact


    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=base_path)


    def objective(trial: optuna.Trial) -> float:
        ... = trial.suggest_float("x", -10, 10)

        # Creating and writing an artifact.
        file_path = generate_example(...)  # This function returns some kind of file.
        artifact_id = upload_artifact(
            trial, file_path, artifact_store
        )  # The return value is the artifact ID.
        trial.set_user_attr(
            "artifact_id", artifact_id
        )  # Save the ID in RDB so that it can be referenced later.

        return ...


    study = optuna.create_study(study_name="test_study", storage="sqlite:///example.db")
    study.optimize(objective, n_trials=100)
    # Loading and displaying artifacts associated with the best trial.
    best_artifact_id = study.best_trial.user_attrs.get("artifact_id")
    with artifact_store.open_reader(best_artifact_id) as f:
        content = f.read().decode("utf-8")

    print(content)

Scenario 2: Remote MySQL RDB server + AWS S3 artifact store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1

    * - Fig 3. Remote MySQL RDB server + AWS S3 artifact store.
    * - .. image:: https://github.com/optuna/optuna/assets/38826298/067efc85-1fad-4b46-a2be-626c64439d7b

Next, we explain the case where data is read and written remotely.

As the scale of optimization increases, it becomes difficult to complete all calculations locally. Optuna's storage objects can 
persist data remotely by specifying a URL, enabling distributed optimization. Here, we will use MySQL as a remote relational 
database server. MySQL is an open-source relational database management system and a well-known software used for various purposes. 
For using MySQL with Optuna, the `tutorial <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html>`_
can be a good reference. However, it is also not appropriate to read and write large data in a relational database like MySQL.

In Optuna, it is common to use the artifact module when you want to read and write such data for each trial. Unlike Scenario 1, 
we distribute the optimization across computation nodes, so  local file system-based backends will not work. Instead, we will use 
AWS S3, an online cloud storage service, and Boto3, a framework for interacting with it from Python. As of v3.3, Optuna has a 
built-in artifact store with this Boto3 backend.

The flow of data is shown in Figure 3. The information calculated in each trial, which corresponds to the optimization history 
(excluding artifact information), is written to the MySQL server. On the other hand, the artifact information is written to AWS S3. 
All workers conducting distributed optimization can read and write in parallel to each, and issues such as race conditions are 
automatically resolved by Optuna's storage module and artifact module. As a result, although the actual data location changes 
between artifact information and non-artifact information (the former is in AWS S3, the latter is in the MySQL RDB), users can 
read and write data transparently. Translating the above process into simple pseudocode would look something like this:

.. code-block:: python

    import os

    import boto3
    from botocore.config import Config
    import optuna
    from optuna.artifact import upload_artifact
    from optuna.artifact.boto3 import Boto3ArtifactStore


    artifact_store = Boto3ArtifactStore(
        client=boto3.client(
            "s3",
            aws_access_key_id=os.environ[
                "PFS2_AWS_ACCESS_KEY_ID"
            ],  # Assume that these environment variables are set up properly. The same applies below.
            aws_secret_access_key=os.environ["PFS2_AWS_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["PFS2_S3_ENDPOINT"],
            config=Config(connect_timeout=30, read_timeout=30),
        ),
        bucket_name=pfs2_bucket,
    )


    def objective(trial: optuna.Trial) -> float:
        ... = trial.suggest_float("x", -10, 10)

        # Creating and writing an artifact.
        file_path = generate_example(...)  # This function returns some kind of file.
        artifact_id = upload_artifact(
            trial, file_path, artifact_store
        )  # The return value is the artifact ID.
        trial.set_user_attr(
            "artifact_id", artifact_id
        )  # Save the ID in RDB so that it can be referenced later.

        return ...


    study = optuna.create_study(
        study_name="test_study",
        storage="mysql://USER:PASS@localhost:3306/test",  # Set the appropriate URL.
    )
    study.optimize(objective, n_trials=100)
    # Loading and displaying artifacts associated with the best trial.
    best_artifact_id = study.best_trial.user_attrs.get("artifact_id")
    with artifact_store.open_reader(best_artifact_id) as f:
        content = f.read().decode("utf-8")

    print(content)

Example: Optimization of Chemical Structures
--------------------------------------------

In this section, we introduce an example of optimizing chemical structure using Optuna by utilizing the artifact module. We will 
target relatively small structures, but the approach remains the same even for complex structures.

Consider the process of a specific molecule adsorbing onto another substance. In this process, the ease of adsorption reaction 
changes depending on the position of the adsorbing molecule to the substance it is adsorbed onto. The ease of adsorption reaction 
can be evaluated by the adsorption energy (the difference between the energy of the system after adsorption and before). By 
formulating the problem as a minimization problem of an objective function that takes the positional relationship of the adsorbing 
molecule as input and outputs the adsorption energy, this problem is solved as a black-box optimization problem.

First, let's import the necessary modules and define some helper functions. You need to install the ASE library for handling 
chemical structures in addition to Optuna, so please install it with `pip install ase`.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import uuid

from ase import Atoms
from ase.build import bulk, fcc111, molecule, add_adsorbate
from ase.calculators.emt import EMT
from ase.io import write, read
from ase.optimize import LBFGS
import numpy as np
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.logging import get_logger
from optuna import create_study
from optuna import Trial


# Add stream handler of stdout to show the messages
get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


def get_opt_energy(atoms: Atoms, fmax: float = 0.001) -> float:
    calculator = EMT()
    atoms.set_calculator(calculator)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax)
    return atoms.get_total_energy()


def create_slab() -> tuple[Atoms, float]:
    calculator = EMT()
    bulk_atoms = bulk("Pt", cubic=True)
    bulk_atoms.calc = calculator

    a = np.mean(np.diag(bulk_atoms.cell))
    slab = fcc111("Pt", a=a, size=(4, 4, 4), vacuum=40.0, periodic=True)
    slab.calc = calculator
    E_slab = get_opt_energy(slab, fmax=1e-4)
    return slab, E_slab


def create_mol() -> tuple[Atoms, float]:
    calculator = EMT()
    mol = molecule("CO")
    mol.calc = calculator
    E_mol = get_opt_energy(mol, fmax=1e-4)
    return mol, E_mol


def atoms_to_json(atoms: Atoms) -> str:
    f = io.StringIO()
    write(f, atoms, format="json")
    return f.getvalue()


def json_to_atoms(atoms_str: str) -> Atoms:
    return read(io.StringIO(atoms_str), format="json")


###################################################################################################
# Each function is as follows.
#
# - `get_opt_energy`: Takes a chemical structure, transitions it to a locally stable structure, and returns the energy after the transition.
# - `create_slab`: Constructs the substance being adsorbed.
# - `create_mol`: Constructs the molecule being adsorbed.
# - `atoms_to_json`: Converts the chemical structure to a string.
# - `json_to_atoms`: Converts the string to a chemical structure.
#
# Using these functions, the code to search for adsorption structures using Optuna is as follows. The objective function is defined
# as class `Objective` in order to carry the artifact store. In its `__call__` method, it retrieves the substance being adsorbed
# (`slab`) and the molecule being adsorbed (`mol`), then after sampling their positional relationship using Optuna (multiple
# `trial.suggest_xxx` methods), it triggers an adsorption reaction with the `add_adsorbate` function, transitions to a locally
# stable structure, then saves the structure in the artifact store and returns the adsorption energy.
#
# The `main` function contains the code to create a `Study` and execute optimization. When creating a `Study`, a storage is
# specified using SQLite, and a back end using the local file system is used for the artifact store. In other words, it corresponds
# to Scenario 1 explained in the previous section. After performing 100 trials of optimization, it displays the information for the
# best trial, and finally saves the chemical structure as `best_atoms.png`. The obtained `best_atoms.png` is shown in Figure 4.


class Objective:
    def __init__(self, artifact_store: FileSystemArtifactStore) -> None:
        self._artifact_store = artifact_store

    def __call__(self, trial: Trial) -> float:
        slab = json_to_atoms(trial.study.user_attrs["slab"])
        E_slab = trial.study.user_attrs["E_slab"]

        mol = json_to_atoms(trial.study.user_attrs["mol"])
        E_mol = trial.study.user_attrs["E_mol"]

        phi = 180.0 * trial.suggest_float("phi", -1, 1)
        theta = np.arccos(trial.suggest_float("theta", -1, 1)) * 180.0 / np.pi
        psi = 180 * trial.suggest_float("psi", -1, 1)
        x_pos = trial.suggest_float("x_pos", 0, 0.5)
        y_pos = trial.suggest_float("y_pos", 0, 0.5)
        z_hig = trial.suggest_float("z_hig", 1, 5)
        xy_position = np.matmul([x_pos, y_pos, 0], slab.cell)[:2]
        mol.euler_rotate(phi=phi, theta=theta, psi=psi)

        add_adsorbate(slab, mol, z_hig, xy_position)
        E_slab_mol = get_opt_energy(slab, fmax=1e-2)

        write(f"./tmp/{trial.number}.json", slab, format="json")
        artifact_id = upload_artifact(trial, f"./tmp/{trial.number}.json", self._artifact_store)
        trial.set_user_attr("structure", artifact_id)

        return E_slab_mol - E_slab - E_mol


def main():
    study = create_study(
        study_name="test_study",
        storage="sqlite:///example.db",
        load_if_exists=True,
    )

    slab, E_slab = create_slab()
    study.set_user_attr("slab", atoms_to_json(slab))
    study.set_user_attr("E_slab", E_slab)

    mol, E_mol = create_mol()
    study.set_user_attr("mol", atoms_to_json(mol))
    study.set_user_attr("E_mol", E_mol)

    os.makedirs("./tmp", exist_ok=True)

    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=base_path)
    study.optimize(Objective(artifact_store), n_trials=3)
    print(
        f"Best trial is #{study.best_trial.number}\n"
        f"    Its adsorption energy is {study.best_value}\n"
        f"    Its adsorption position is\n"
        f"        phi  : {study.best_params['phi']}\n"
        f"        theta: {study.best_params['theta']}\n"
        f"        psi. : {study.best_params['psi']}\n"
        f"        x_pos: {study.best_params['x_pos']}\n"
        f"        y_pos: {study.best_params['y_pos']}\n"
        f"        z_hig: {study.best_params['z_hig']}"
    )

    best_artifact_id = study.best_trial.user_attrs["structure"]
    with artifact_store.open_reader(best_artifact_id) as f:
        content = f.read().decode("utf-8")
    best_atoms = json_to_atoms(content)
    print(best_atoms)
    write("best_atoms.png", best_atoms, rotation=("315x,0y,0z"))


if __name__ == "__main__":
    main()

###################################################################################################
# .. list-table::
#     :header-rows: 1
#
#    * - Fig 4. The chemical structure obtained by the above code.
#    * - .. image:: https://github.com/optuna/optuna/assets/38826298/c6bd62fd-599a-424e-8c2c-ca88af85cc63
#
# As shown above, it is convenient to use the artifact module when performing the optimization of chemical structures with Optuna.
# In the case of small structures or fewer trial numbers, it's fine to convert it to a string and save it directly in the RDB.
# However, when dealing with complex structures or performing large-scale searches, it's better to save it outside the RDB to
# avoid overloading it, such as in an external file system or AWS S3.
#
# Conclusion
# ----------
#
# The artifact module is a useful feature when you want to save relatively large data for each trial. It can be used for various
# purposes such as saving snapshots of machine learning models, optimizing chemical structures, and human-in-the-loop optimization
# of images and sounds. It's a powerful assistant for black-box optimization with Optuna. Also, if there are ways to use it that
# we, the Optuna committers, haven't noticed, please let us know on GitHub discussions. Have a great optimization life with Optuna!
