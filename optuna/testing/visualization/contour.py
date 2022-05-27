import abc
from enum import Enum
from io import BytesIO
import tempfile
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization._plotly_imports import go
from optuna.visualization.matplotlib import plot_contour as matplotlib_plot_contour
from optuna.visualization.matplotlib._matplotlib_imports import Axes


class VarType(Enum):
    UNIFORM = "uniform"
    LOG = "log"
    CATEGORICAL = "category"


class BaseContourFigure(object, metaclass=abc.ABCMeta):
    """Base class of the contour figure.

    If `n_params` == `d`, there are `d * (d - 1)` figures. The exception to this rule is when
    `d` = 2. If `d` = 2, there is only one figure.

    All figures should be accessed by its number. If `d` = 4, the number of each figure is defined
    as follows.
    param_1 |         | plot_0  | plot_1  | plot_2  |
    param_2 | plot_3  |         | plot_4  | plot_5  |
    param_3 | plot_6  | plot_7  |         | plot_8  |
    param_4 | plot_9  | plot_10 | plot_11 |         |
              param_1   param_2   param_3   param_4
    """

    def __init__(self, study: Study, params: Optional[List[str]] = None) -> None:
        if len(study.get_trials(states=(TrialState.COMPLETE,))) == 0:
            return

        if params is not None and len(params) < 2:
            return

        self.params = params if params is not None else list(study.trials[0].params.keys())
        self.params.sort()
        self.n_params = len(self.params)

    def get_n_params(self) -> int:
        return len(self.params)

    def get_param_names(self) -> List[str]:
        return self.params

    def get_n_plots(self) -> int:
        if self.get_n_params() == 2:
            return 1
        return self.get_n_params() * (self.get_n_params() - 1)

    @abc.abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def save_static_image(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_grid(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_grid(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_z_map(self, n: int = 0) -> Optional[List[List[float]]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_points(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_points(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_range(self, n: int = 0) -> Tuple[float, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_range(self, n: int = 0) -> Tuple[float, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_name(self, n: int = 0) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_name(self, n: int = 0) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_target_name(self) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_type(self, n: int = 0) -> VarType:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_type(self, n: int = 0) -> VarType:
        raise NotImplementedError


class PlotlyContourFigure(BaseContourFigure):
    def __init__(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
        target_name: str = "Objective Value",
    ) -> None:
        super().__init__(study, params)

        figure = plot_contour(study, params=params, target=target, target_name=target_name)

        self.figure = figure

        # self.plots[i] includes
        #   "scatter": go.Scatter
        #   "contour": go.Contour
        #   "xaxis"  : go.layout.XAxis
        #   "yaxis"  : go.layout.YAxis
        #   "xname"  : str
        #   "yname"  : str
        self.plots = [{} for _ in range(self.get_n_plots())]

        # Used if `n_params` > 2.
        _removed_axis_numbers = [k * self.n_params + k + 1 for k in range(self.n_params)]

        def _axis_number_to_plot_number(x: int) -> int:
            n_removed_less_than_x = sum([1 for y in _removed_axis_numbers if y < x])
            return x - n_removed_less_than_x - 1

        for d in figure.data:
            if isinstance(d, go.Scatter):
                if self.n_params == 2:
                    plot_number = 0
                else:
                    axis_number = int(d.xaxis[1:]) if d.xaxis != "x" else 1
                    if axis_number in _removed_axis_numbers:
                        continue
                    plot_number = _axis_number_to_plot_number(axis_number)
                self.plots[plot_number]["scatter"] = d
            elif isinstance(d, go.Contour):
                if self.n_params == 2:
                    plot_number = 0
                else:
                    axis_number = int(d.xaxis[1:]) if d.xaxis != "x" else 1
                    plot_number = _axis_number_to_plot_number(axis_number)
                self.plots[plot_number]["contour"] = d
            else:
                assert False

        def _create_axis_by_name(layout_key: str, axis_name: str) -> None:
            # Set axis.
            if self.n_params == 2:
                plot_number = 0
            else:
                axis_number = int(layout_key[5:]) if layout_key != axis_name else 1
                if axis_number in _removed_axis_numbers:
                    return
                plot_number = _axis_number_to_plot_number(axis_number)
            self.plots[plot_number][axis_name] = figure.layout[layout_key]

            # Set parameter name for the axis.
            if figure.layout[layout_key].title.text is not None:
                param_name = figure.layout[layout_key].title.text
            else:
                # The figure.layout[layout_key].matches returns the mathched axis like `x12`.
                # We need to convert it to the actual axis name like `xaxis12`.
                matched_axis = figure.layout[layout_key].matches
                axis_name = (
                    axis_name + matched_axis[1:] if matched_axis != axis_name[0] else axis_name
                )
                param_name = figure.layout[axis_name].title.text
            self.plots[plot_number][axis_name[0] + "name"] = param_name

        for l in figure.layout:
            if l.startswith("xaxis"):
                _create_axis_by_name(l, "xaxis")
            elif l.startswith("yaxis"):
                _create_axis_by_name(l, "yaxis")

    def is_empty(self) -> bool:
        return len(self.figure.data) == 0

    def save_static_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.figure.write_image(td + "tmp.png")

    def get_x_grid(self, n: int = 0) -> Optional[List[float]]:
        if self.plots[n]["contour"].x is None:
            return None
        return list(self.plots[n]["contour"].x)

    def get_y_grid(self, n: int = 0) -> Optional[List[float]]:
        if self.plots[n]["contour"].y is None:
            return None
        return list(self.plots[n]["contour"].y)

    def get_z_map(self, n: int = 0) -> Optional[List[List[float]]]:
        if self.plots[n]["contour"].z is None:
            return None
        return list(self.plots[n]["contour"].z)

    def get_x_points(self, n: int = 0) -> Optional[List[float]]:
        if self.plots[n]["scatter"].x is None:
            return None
        return list(self.plots[n]["scatter"].x)

    def get_y_points(self, n: int = 0) -> Optional[List[float]]:
        if self.plots[n]["scatter"].y is None:
            return None
        return list(self.plots[n]["scatter"].y)

    def get_x_range(self, n: int = 0) -> Tuple[float, float]:
        return self.plots[n]["xaxis"].range

    def get_y_range(self, n: int = 0) -> Tuple[float, float]:
        return self.plots[n]["yaxis"].range

    def _get_titles(self, start_string: str) -> List[str]:
        titles = []
        for l in self.figure.layout:
            if l.startswith(start_string) and self.figure.layout[l].title.text is not None:
                titles.append(self.figure.layout[l].title.text)
        titles.sort()
        return titles

    def get_x_name(self, n: int = 0) -> str:
        return self.plots[n]["xname"]

    def get_y_name(self, n: int = 0) -> str:
        return self.plots[n]["yname"]

    def get_target_name(self) -> Optional[str]:
        return self.plots[0]["contour"].colorbar.title.text

    def _get_type_by_name(self, n: int, name: str) -> VarType:
        var_type = self.plots[n][name].type
        if var_type is None:
            return VarType.UNIFORM
        elif var_type == "log":
            return VarType.LOG
        elif var_type == "category":
            return VarType.CATEGORICAL
        else:
            assert False

    def get_x_type(self, n: int = 0) -> VarType:
        return self._get_type_by_name(n, "xaxis")

    def get_y_type(self, n: int = 0) -> VarType:
        return self._get_type_by_name(n, "yaxis")


class MatplotlibContourFigure(BaseContourFigure):
    def __init__(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
        target_name: str = "Objective Value",
    ) -> None:
        super().__init__(study, params)

        figure = matplotlib_plot_contour(
            study, params=params, target=target, target_name=target_name
        )

        self.figure = figure

        # self.plots[i] has a type of Axes.
        self.plots = [Axes() for _ in range(self.get_n_plots())]

        if isinstance(figure, Axes):
            self.plots[0] = figure
        elif isinstance(figure, np.ndarray):
            _removed_axis_numbers = [k * self.n_params + k for k in range(self.n_params)]

            def _axis_number_to_plot_number(x: int) -> int:
                n_removed_less_than_x = sum([1 for y in _removed_axis_numbers if y < x])
                return x - n_removed_less_than_x

            for axis_number, f in enumerate(figure.flatten()):
                f.set_xlabel(figure[-1][axis_number % self.n_params].get_xlabel())
                f.set_ylabel(figure[axis_number / self.n_params][0].get_ylabel())
                if axis_number in _removed_axis_numbers:
                    assert not f.has_data()
                    continue
                f.set_xticklabels(figure[-1][axis_number % self.n_params].get_xticklabels())
                f.set_yticklabels(figure[axis_number / self.n_params][0].get_yticklabels())
                plot_number = _axis_number_to_plot_number(axis_number)
                assert f.has_data()
                self.plots[plot_number] = f
        else:
            assert False

    def is_empty(self) -> bool:
        return not all([f.has_data() for f in self.plots])

    def save_static_image(self) -> None:
        plt.savefig(BytesIO())

    def get_x_grid(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    def get_y_grid(self, n: int = 0) -> Optional[List[float]]:
        raise NotImplementedError

    def get_z_map(self, n: int = 0) -> Optional[List[List[float]]]:
        raise NotImplementedError

    def get_x_points(self, n: int = 0) -> Optional[List[float]]:
        if len(self.plots[n].collections) == 0:
            return None
        return list(self.plots[n].collections[-1].get_offsets()[0])

    def get_y_points(self, n: int = 0) -> Optional[List[float]]:
        if len(self.plots[n].collections) == 0:
            return None
        return list(self.plots[n].collections[-1].get_offsets()[1])

    def get_x_range(self, n: int = 0) -> Tuple[float, float]:
        return self.plots[n].get_xlim()

    def get_y_range(self, n: int = 0) -> Tuple[float, float]:
        return self.plots[n].get_ylim()

    def get_x_name(self, n: int = 0) -> str:
        return self.plots[n].get_xlabel()

    def get_y_name(self, n: int = 0) -> str:
        return self.plots[n].get_ylabel()

    def get_target_name(self) -> Optional[str]:
        raise self.plots[0].figure.axes[-1].get_ylabel()

    def get_x_type(self, n: int = 0) -> VarType:
        var_type = self.plots[n].get_xscale()
        if var_type == "linear" and not self.is_category[n][0]:
            return VarType.UNIFORM
        elif var_type == "log":
            return VarType.LOG
        elif var_type == "linear" and self.is_category[n][0]:
            return VarType.CATEGORICAL
        else:
            assert False

    def get_y_type(self, n: int = 0) -> VarType:
        var_type = self.plots[n].get_yscale()
        if var_type == "linear" and not self.is_category[n][1]:
            return VarType.UNIFORM
        elif var_type == "log":
            return VarType.LOG
        elif var_type == "linear" and self.is_category[n][1]:
            return VarType.CATEGORICAL
        else:
            assert False
