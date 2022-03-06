import math
from kurobako import problem
from WFGtestSuite import wfg


class WFGProblemFactory(problem.ProblemFactory):
    def specification(self):
        params = [
            problem.Var("x", problem.ContinuousRange(0, 2)),
            problem.Var("y", problem.ContinuousRange(0, 4)),
        ]
        return problem.ProblemSpec(
            name="WFG2",
            params=params,
            values=[problem.Var("f1"), problem.Var("f2")],
        )

    def create_problem(self, seed):
        return WFGProblem()


class WFGProblem(problem.Problem):
    def create_evaluator(self, params):
        return WFGEvaluator(params)


class WFGEvaluator(problem.Evaluator):
    def __init__(self, params):
        self._x, self._y = params
        self._current_step = 0
        self.wfg = wfg.WFG2(n_arguments=2, n_objectives=2, k=1)

    def current_step(self):
        return self._current_step

    def evaluate(self, next_step):
        self._current_step = 1
        x, y = self._x, self._y
        v = self.wfg([x, y])
        v = v.tolist()

        if math.isnan(v[0]) or math.isinf(v[0]):
            raise ValueError
        if math.isnan(v[1]) or math.isinf(v[1]):
            raise ValueError
        return v


if __name__ == "__main__":
    runner = problem.ProblemRunner(WFGProblemFactory())
    runner.run()
