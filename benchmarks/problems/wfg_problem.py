import math
import sys

from kurobako import problem

from WFGtestSuite import wfg


class WFGProblemFactory(problem.ProblemFactory):
    def specification(self):
        self._n_wfg = int(sys.argv[1])
        self._n_dim = int(sys.argv[2])

        self._low = 0
        self._high = 2
        params = [
            problem.Var(f"x{i}", problem.ContinuousRange(0, self._high * i))
            for i in range(self._n_dim)
        ]
        return problem.ProblemSpec(
            name=f"WFG{self._n_wfg}",
            params=params,
            values=[problem.Var("f1"), problem.Var("f2")],
        )

    def create_problem(self, seed):
        return WFGProblem()


class WFGProblem(problem.Problem):
    def __init__(self) -> None:
        super().__init__()

    def create_evaluator(self, params):
        return WFGEvaluator(params)


class WFGEvaluator(problem.Evaluator):
    def __init__(self, params):
        self._n_wfg = int(sys.argv[1])
        self._n_dim = int(sys.argv[2])
        self._n_obj = int(sys.argv[3])
        self._k = int(sys.argv[4])

        self._x = params
        self._current_step = 0

        if self._n_wfg == 1:
            self.wfg = wfg.WFG1(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 2:
            self.wfg = wfg.WFG2(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 3:
            self.wfg = wfg.WFG3(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 4:
            self.wfg = wfg.WFG4(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 5:
            self.wfg = wfg.WFG5(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 6:
            self.wfg = wfg.WFG6(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 7:
            self.wfg = wfg.WFG7(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 8:
            self.wfg = wfg.WFG8(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 9:
            self.wfg = wfg.WFG9(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        else:
            assert False, "Invalid specification for WFG number."

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
