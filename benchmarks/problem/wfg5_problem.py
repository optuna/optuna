import math

from kurobako import problem

from WFGtestSuite import wfg

N_VAR = 3
K = 2


class WFGProblemFactory(problem.ProblemFactory):
    def specification(self):
        self._low = 0
        self._high = 2
        params = [
            problem.Var(f"x{i}", problem.ContinuousRange(0, self._high * i)) for i in range(N_VAR)
        ]
        return problem.ProblemSpec(
            name="WFG5",
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
        self._x = params
        self._current_step = 0
        self.wfg = wfg.WFG5(n_arguments=N_VAR, n_objectives=2, k=K)

    def current_step(self):
        return self._current_step

    def evaluate(self, next_step):
        self._current_step = 1
        v = self.wfg(self._x)
        v = v.tolist()

        if math.isnan(v[0]) or math.isinf(v[0]):
            raise ValueError
        if math.isnan(v[1]) or math.isinf(v[1]):
            raise ValueError
        return v


if __name__ == "__main__":
    runner = problem.ProblemRunner(WFGProblemFactory())
    runner.run()
