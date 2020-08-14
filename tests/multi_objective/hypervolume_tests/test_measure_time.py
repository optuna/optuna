from ast import literal_eval
import csv
import time

import numpy as np

import optuna


def test_file_wfg_2d() -> None:
    with open('/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_int_wfg_2d.csv') as f:
        with open('/Users/mamu/Documents/Work/pfn/exercise_pagmo/comp_time_2d_without_e2d.csv', 'w') as g:
            reader = csv.reader(f, delimiter=' ')
            for i, row in enumerate(reader):
                if (i + 1) // 100 == 0:
                    if (i + 1) % 10 != 0:
                        continue
                elif (i + 1) % 100 != 0:
                    continue
                s = np.asarray(literal_eval(row[0]))
                r = np.asarray(literal_eval(row[1]))
                t = float(row[3]) / 1e+6
                start = time.time()
                optuna.multi_objective._hypervolume.WFG().compute(s, r)
                elapsed = time.time() - start
                g.write("{} {} {}\n".format(i + 1, t, elapsed))


def test_file_wfg_3d() -> None:
    with open("/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_int_wfg_3d.csv") as f:
        with open('/Users/mamu/Documents/Work/pfn/exercise_pagmo/comp_time_3d.csv', 'w') as g:
            reader = csv.reader(f, delimiter=" ")
            for i, row in enumerate(reader):
                if (i + 1) // 100 == 0:
                    if (i + 1) % 10 != 0:
                        continue
                elif (i + 1) % 100 != 0:
                    continue
                s = np.asarray(literal_eval(row[0]))
                r = np.asarray(literal_eval(row[1]))
                t = float(row[3]) / 1e+6
                start = time.time()
                optuna.multi_objective._hypervolume.WFG().compute(s, r)
                elapsed = time.time() - start
                g.write("{} {} {}\n".format(i + 1, t, elapsed))


def test_file_wfg_10d() -> None:
    with open("/Users/mamu/Documents/Work/pfn/exercise_pagmo/test_cases_int_wfg_10d.csv") as f:
        with open('/Users/mamu/Documents/Work/pfn/exercise_pagmo/comp_time_10d.csv', 'w') as g:
            reader = csv.reader(f, delimiter=" ")
            for i, row in enumerate(reader):
                if (i + 1) // 100 == 0:
                    if (i + 1) % 10 != 0:
                        continue
                elif (i + 1) % 100 != 0:
                    continue
                s = np.asarray(literal_eval(row[0]))
                r = np.asarray(literal_eval(row[1]))
                t = float(row[3]) / 1e+6
                start = time.time()
                optuna.multi_objective._hypervolume.WFG().compute(s, r)
                elapsed = time.time() - start
                g.write("{} {} {}\n".format(i + 1, t, elapsed))
