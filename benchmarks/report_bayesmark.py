from collections import Counter
import io
import json
import os
from typing import Dict


_TABLE_HEADER = "\n|Ranking|Solver|Score|\n|---:|:---|---:|\n"


class BayesmarkReportBuilder:
    def __init__(self) -> None:

        self._solvers: Counter = Counter()
        self._datasets: Counter = Counter()
        self._models: Counter = Counter()
        self._problems_counter = 1
        self._problems_body = io.StringIO()

    def add_problem(self, name: str, scores: Dict[str, float]) -> "BayesmarkReportBuilder":

        if self._problems_body.closed:
            self._problems_body = io.StringIO()

        problem_title = f"### ({self._problems_counter}) Problem: {name}\n"
        self._problems_body.write(problem_title)
        self._problems_body.write(_TABLE_HEADER)

        for idx, (solver, score) in enumerate(scores.items()):
            row = f"|{idx + 1}|{solver}|{score:.5f}|\n"
            self._problems_body.write(row)

        self._solvers.update(scores.keys())
        self._problems_counter += 1

        return self

    def add_dataset(self, dataset: str) -> "BayesmarkReportBuilder":

        self._datasets.update([dataset])
        return self

    def add_model(self, model: str) -> "BayesmarkReportBuilder":

        self._models.update([model])
        return self

    def assemble_report(self) -> str:

        num_datasets = len(self._datasets)
        num_models = len(self._models)

        with open(os.path.join("benchmarks", "bayesmark", "report_template.md")) as file:
            report_template = file.read()

        # TODO(xadrianzetx) Consider using proper templating engine.
        report = report_template.format(
            num_solvers=len(self._solvers),
            num_datasets=num_datasets,
            num_models=num_models,
            num_problems=num_datasets * num_models,
            leaderboards=self._problems_body.getvalue(),
        )

        self._problems_body.close()
        return report


def build_report() -> None:

    report_builder = BayesmarkReportBuilder()
    for partial_name in os.listdir("partial"):
        dataset, model, *_ = partial_name.split("-")
        path = os.path.join("partial", partial_name)

        with open(path) as file:
            scores = json.load(file)
            problem_name = f"{dataset.capitalize()}-{model}"
            report_builder = (
                report_builder.add_problem(problem_name, scores)
                .add_dataset(dataset)
                .add_model(model)
            )

    report = report_builder.assemble_report()
    with open(os.path.join("report", "benchmark-report.md"), "w") as file:
        file.write(report)


if __name__ == "__main__":
    os.makedirs("report", exist_ok=True)
    build_report()
