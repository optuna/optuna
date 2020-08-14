from ast import literal_eval
import csv

import numpy as np

import optuna


def test_file_wfg_2d() -> None:
    with open('/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_int_wfg_2d.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            s = np.asarray(literal_eval(row[0]))
            r = np.asarray(literal_eval(row[1]))
            v = float(row[2])
            assert np.isclose(v, optuna.multi_objective._hypervolume.WFG().compute(s, r), rtol=1e-2)


def test_file_wfg_3d() -> None:
    with open("/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_int_wfg_3d.csv") as f:
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            if i > 100:
                continue
            s = np.asarray(literal_eval(row[0]))
            r = np.asarray(literal_eval(row[1]))
            v = float(row[2])
            assert np.isclose(v, optuna.multi_objective._hypervolume.WFG().compute(s, r), rtol=1e-2)


def test_file_wfg_10d() -> None:
    with open("/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_wfg.csv") as f:
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            if i > 20:
                continue
            s = np.asarray(literal_eval(row[0]))
            r = np.asarray(literal_eval(row[1]))
            v = float(row[2])
            assert np.isclose(v, optuna.multi_objective._hypervolume.WFG().compute(s, r), rtol=1e-2)
