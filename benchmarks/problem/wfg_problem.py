from kurobako import problem


class WSGProblemFactory(problem.ProblemFactory):
    def specification(self):
        # wip
        params = []

        return problem.ProblemSpec(name="WFG", params=params, vales=[])

    def create_problem(self, seed):
        return WSGProblem()


class WSGProblem(problem.Problem):
    def create_evaluator(self, params):
        return WSGProblem(params)


class WSGEvaluator(problem.Evaluator):
    def __init__(self, params):
        self._x, self._y = params
        self._current_step = 0

    def current_step(self):
        return self._current_step

    def evaluate(self, next_step):
        self._current_step = 1

        # TODO:
        return None


if __name__ == "__main__":
    runner = problem.ProblemRunner(WSGProblemFactory())
    runner.run()
