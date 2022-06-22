import abc
from collections import defaultdict
import io
import itertools
import os
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import mannwhitneyu


_LINE_BREAK = "\n"
_TABLE_HEADER = "|Ranking|Solver|"
_HEADER_FORMAT = "|:---|---:|"
_OVERALL_HEADER = "|Solver|Borda|Firsts|\n|:---|---:|---:|\n"

Moments = Tuple[float, float]
Samples = Dict[str, List[float]]


class BaseMetric(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Metric name displayed in final report."""

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def precision(self) -> int:
        """Number of digits following decimal point displayed in final report."""

        raise NotImplementedError

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame) -> List[float]:
        """Calculates metric for each study in data frame."""

        raise NotImplementedError


class BestValueMetric(BaseMetric):
    name = "Best value"
    precision = 6

    def calculate(self, data: pd.DataFrame) -> List[float]:

        return data.groupby("uuid").generalization.min().values


class AUCMetric(BaseMetric):
    name = "AUC"
    precision = 3

    def calculate(self, data: pd.DataFrame) -> List[float]:

        aucs: List[float] = list()
        for _, grp in data.groupby("uuid"):
            auc = np.sum(grp.generalization.cummin())
            aucs.append(auc / grp.shape[0])
        return aucs


class ElapsedMetric(BaseMetric):
    name = "Elapsed"
    precision = 3

    def calculate(self, data: pd.DataFrame) -> List[float]:

        # Total time does not include evaluation of bayesmark
        # objective function (no Optuna APIs are called there).
        time_cols = ["suggest", "observe"]
        return data.groupby("uuid")[time_cols].sum().sum(axis=1).values


class PartialReport:
    def __init__(self, data: pd.DataFrame) -> None:

        self._data = data

    @property
    def optimizers(self) -> List[str]:

        return list(self._data.opt.unique())

    @classmethod
    def from_json(cls, path: str) -> "PartialReport":

        data = pd.read_json(path)
        return cls(data)

    def summarize_solver(self, solver: str, metric: BaseMetric) -> Moments:

        solver_data = self._data[self._data.opt == solver]
        if solver_data.shape[0] == 0:
            raise ValueError(f"{solver} not found in report.")

        run_metrics = metric.calculate(solver_data)
        return np.mean(run_metrics), np.var(run_metrics)

    def sample_performance(self, metric: BaseMetric) -> Samples:

        performance: Dict[str, List[float]] = {}
        for solver, data in self._data.groupby("opt"):
            run_metrics = metric.calculate(data)
            performance[solver] = run_metrics
        return performance


class DewanckerRanker:
    def __init__(self, metrics: List[BaseMetric]) -> None:
        self._metrics = metrics
        self._ranking: Optional[List[str]] = None
        self._borda: Optional[np.ndarray] = None

    def __iter__(self) -> Generator[Tuple[str, int], None, None]:

        yield from zip(self.solvers, self.borda)

    @property
    def solvers(self) -> List[str]:

        if self._ranking is None:
            raise ValueError("Call rank first.")
        return self._ranking

    @property
    def borda(self) -> np.ndarray:

        if self._borda is None:
            raise ValueError("Call rank first.")
        return self._borda

    @staticmethod
    def pick_alpha(report: PartialReport) -> float:

        # https://github.com/optuna/kurobako/blob/788dd4cf618965a4a5158aa4e13607a5803dea9d/src/report.rs#L412-L424  # noqa E501
        num_optimizers = len(report.optimizers)
        candidates = [0.075, 0.05, 0.025, 0.01] * 4 / np.repeat([1, 10, 100, 1000], 4)

        for cand in candidates:
            if 1 - np.power((1 - cand), binom(num_optimizers, 2)) < 0.05:
                return cand
        return candidates[-1]

    def _set_ranking(self, wins: Dict[str, int]) -> None:

        sorted_wins = [k for k, _ in sorted(wins.items(), key=lambda x: x[1])]
        self._ranking = sorted_wins[::-1]

    def _set_borda(self, wins: Dict[str, int]) -> None:

        sorted_wins = np.array(sorted(wins.values()))
        num_wins, num_ties = np.unique(sorted_wins, return_counts=True)
        points = np.searchsorted(sorted_wins, num_wins)
        self._borda = np.repeat(points, num_ties)[::-1]

    def rank(self, report: PartialReport) -> None:

        # Implements Section 2.1.1
        # https://proceedings.mlr.press/v64/dewancker_strategy_2016.pdf
        wins: Dict[str, int] = defaultdict(int)
        alpha = DewanckerRanker.pick_alpha(report)
        for metric in self._metrics:
            samples = report.sample_performance(metric)
            for optim_a, optim_b in itertools.permutations(samples, 2):
                _, p_val = mannwhitneyu(samples[optim_a], samples[optim_b], alternative="less")
                if p_val < alpha:
                    wins[optim_a] += 1

            all_wins = [wins[optimizer] for optimizer in report.optimizers]
            no_ties = len(all_wins) == len(np.unique(all_wins))
            if no_ties:
                break

        wins = {optimzier: wins[optimzier] for optimzier in report.optimizers}
        self._set_ranking(wins)
        self._set_borda(wins)


class BayesmarkReportBuilder:
    def __init__(self) -> None:

        self._solvers: Set[str] = set()
        self._datasets: Set[str] = set()
        self._models: Set[str] = set()
        self._firsts: Dict[str, int] = defaultdict(int)
        self._borda: Dict[str, int] = defaultdict(int)
        self._metric_precedence = str()
        self._problems_counter = 1
        self._problems_body = io.StringIO()

    def set_precedence(self, metrics: List[BaseMetric]) -> None:

        self._metric_precedence = " -> ".join([m.name for m in metrics])

    def add_problem(
        self,
        name: str,
        report: PartialReport,
        ranking: DewanckerRanker,
        metrics: List[BaseMetric],
    ) -> "BayesmarkReportBuilder":

        if self._problems_body.closed:
            self._problems_body = io.StringIO()

        problem_header = f"### ({self._problems_counter}) Problem: {name}" + _LINE_BREAK
        self._problems_body.write(problem_header)
        metric_names = [f"{m.name} (avg +- std) |" for m in metrics]
        self._problems_body.write(
            "".join([_LINE_BREAK, _TABLE_HEADER, *metric_names, _LINE_BREAK])
        )
        metric_format = ["---:|" for _ in range(len(metrics))]
        self._problems_body.write("".join([_HEADER_FORMAT, *metric_format, _LINE_BREAK]))

        positions = np.abs(ranking.borda - (max(ranking.borda) + 1))
        for pos, solver in zip(positions, ranking.solvers):
            self._solvers.add(solver)
            row_buffer = io.StringIO()
            row_buffer.write(f"|{pos}|{solver}|")
            for metric in metrics:
                mean, variance = report.summarize_solver(solver, metric)
                precision = metric.precision
                row_buffer.write(f"{mean:.{precision}f} +- {np.sqrt(variance):.{precision}f}|")

            self._problems_body.write("".join([row_buffer.getvalue(), _LINE_BREAK]))
            row_buffer.close()

        self._problems_counter += 1
        return self

    def update_leaderboard(self, ranking: DewanckerRanker) -> "BayesmarkReportBuilder":

        for solver, borda in ranking:
            if borda == max(ranking.borda):
                self._firsts[solver] += 1
            self._borda[solver] += borda
        return self

    def add_dataset(self, dataset: str) -> "BayesmarkReportBuilder":

        self._datasets.update(dataset)
        return self

    def add_model(self, model: str) -> "BayesmarkReportBuilder":

        self._models.update(model)
        return self

    def assemble_report(self) -> str:

        num_datasets = len(self._datasets)
        num_models = len(self._models)

        overall_body = io.StringIO()
        overall_body.write(_OVERALL_HEADER)
        for solver in self._solvers:
            row = f"|{solver}|{self._borda[solver]}|{self._firsts[solver]}|"
            overall_body.write("".join([row, _LINE_BREAK]))

        with open(os.path.join("benchmarks", "bayesmark", "report_template.md")) as file:
            report_template = file.read()

        # TODO(xadrianzetx) Consider using proper templating engine.
        report = report_template.format(
            num_solvers=len(self._solvers),
            num_datasets=num_datasets,
            num_models=num_models,
            precedence=self._metric_precedence,
            num_problems=num_datasets * num_models,
            overall=overall_body.getvalue(),
            leaderboards=self._problems_body.getvalue(),
        )

        overall_body.close()
        self._problems_body.close()
        return report


def build_report() -> None:

    # Order of this list sets metric precedence.
    # https://proceedings.mlr.press/v64/dewancker_strategy_2016.pdf
    metrics = [BestValueMetric(), AUCMetric()]
    report_builder = BayesmarkReportBuilder()
    report_builder.set_precedence(metrics)

    for partial_name in os.listdir("partial"):
        dataset, model, *_ = partial_name.split("-")
        problem_name = f"{dataset.capitalize()}-{model}"
        path = os.path.join("partial", partial_name)

        partial = PartialReport.from_json(path)
        ranking = DewanckerRanker(metrics)
        ranking.rank(partial)

        # Elapsed time is not used as a voting metric, but shown in report
        # so it gets added to metric pool *after* ranking was calculated.
        elapsed = ElapsedMetric()
        all_metrics = [*metrics, elapsed]

        report_builder = (
            report_builder.add_problem(problem_name, partial, ranking, all_metrics)
            .add_dataset(dataset)
            .add_model(model)
            .update_leaderboard(ranking)
        )

    report = report_builder.assemble_report()
    with open(os.path.join("report", "benchmark-report.md"), "w") as file:
        file.write(report)


if __name__ == "__main__":
    os.makedirs("report", exist_ok=True)
    build_report()
