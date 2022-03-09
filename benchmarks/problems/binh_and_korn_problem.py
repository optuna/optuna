from kurobako import problem


class BinhAndKornProblemFactory(problem.ProblemFactory):
    def specification(self):
        params = [
            problem.Var("x", problem.ContinuousRange(0, 5)),
            problem.Var("y", problem.ContinuousRange(0, 3)),
        ]
        return problem.ProblemSpec(
            name="BinhKorn",
            params=params,
            values=[
                problem.Var("4 * x ** 2 + 4 * y ** 2"),
                problem.Var("(x - 5) ** 2 + (y - 5) ** 2"),
            ],
        )

    def create_problem(self, seed):
        return BinhAndKornProblem()


class BinhAndKornProblem(problem.Problem):
    def create_evaluator(self, params):
        return BinhAndKornEvaluator(params)


class BinhAndKornEvaluator(problem.Evaluator):
    def __init__(self, params):
        self._x, self._y = params
        self._current_step = 0

    def current_step(self):
        return self._current_step

    def evaluate(self, next_step):
        self._current_step = 1
        x, y = self._x, self._y
        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return [v0, v1]


if __name__ == "__main__":
    runner = problem.ProblemRunner(BinhAndKornProblemFactory())
    runner.run()
