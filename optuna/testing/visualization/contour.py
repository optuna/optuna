import abc
from io import BytesIO
import tempfile
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np

from optuna.distributions import CategoricalChoiceType
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization.matplotlib import plot_contour as plt_plot_contour


class BaseTestableContourFigure(object, metaclass=abc.ABCMeta):
    """Base class of the testable contour figure.

    The methods except the followings are only used when `n_params = 2`.
    - `get_n_params`
    - `get_n_plots`
    - `is_empty`
    - `save_static_image`
    """

    def __init__(self, study: Study, params: Optional[List[str]] = None) -> None:
        if len(study.get_trials(states=(TrialState.COMPLETE,))) == 0:
            return

        if params is not None and len(params) < 2:
            return

        params = params if params is not None else list(study.trials[0].params.keys())
        self.n_params = len(params)

    def get_n_params(self) -> int:
        return self.n_params

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
    def get_x_points(self) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_points(self) -> Optional[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_range(self) -> Tuple[float, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_range(self) -> Tuple[float, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_ticks(self) -> List[CategoricalChoiceType]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_ticks(self) -> List[CategoricalChoiceType]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_target_name(self) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_is_log(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_is_log(self) -> bool:
        raise NotImplementedError


class PlotlyContourFigure(BaseTestableContourFigure):
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

        if self.is_empty() or len(figure.data) > 2:
            return

        self.contour = figure.data[0]
        self.scatter = figure.data[1]
        self.layout = figure.layout

    def is_empty(self) -> bool:
        return len(self.figure.data) == 0

    def save_static_image(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.figure.write_image(td + "tmp.png")

    def get_x_points(self) -> Optional[List[float]]:
        return list(self.scatter["x"]) if self.scatter["x"] is not None else None

    def get_y_points(self) -> Optional[List[float]]:
        return list(self.scatter["y"]) if self.scatter["y"] is not None else None

    def get_x_range(self) -> Tuple[float, float]:
        return self.layout["xaxis"].range

    def get_y_range(self) -> Tuple[float, float]:
        return self.layout["yaxis"].range

    def get_x_name(self) -> str:
        return self.layout["xaxis"].title.text

    def get_y_name(self) -> str:
        return self.layout["yaxis"].title.text

    def get_x_ticks(self) -> List[CategoricalChoiceType]:
        return list(self.contour["x"])

    def get_y_ticks(self) -> List[CategoricalChoiceType]:
        return list(self.contour["y"])

    def get_target_name(self) -> Optional[str]:
        return self.contour["colorbar"].title.text

    def get_x_is_log(self) -> bool:
        return self.layout["xaxis"].type == "log"

    def get_y_is_log(self) -> bool:
        return self.layout["yaxis"].type == "log"


class MatplotlibContourFigure(BaseTestableContourFigure):
    def __init__(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
        target_name: str = "Objective Value",
    ) -> None:
        super().__init__(study, params)
        figure = plt_plot_contour(study, params=params, target=target, target_name=target_name)
        self.figure = figure

    def is_empty(self) -> bool:
        if isinstance(self.figure, Axes):
            return not self.figure.has_data()
        else:
            return not any([f.has_data() for f in self.figure.flatten()])

    def save_static_image(self) -> None:
        plt.savefig(BytesIO())

    def get_x_points(self) -> Optional[List[float]]:
        if len(self.figure.collections) == 0:
            return None
        return list(self.figure.collections[-1].get_offsets()[:, 0])

    def get_y_points(self) -> Optional[List[float]]:
        if len(self.figure.collections) == 0:
            return None
        return list(self.figure.collections[-1].get_offsets()[:, 1])

    def get_x_range(self) -> Tuple[float, float]:
        if self.get_x_is_log():
            l, r = self.figure.get_xlim()
            return np.log10(l), np.log10(r)
        return self.figure.get_xlim()

    def get_y_range(self) -> Tuple[float, float]:
        if self.get_y_is_log():
            l, r = self.figure.get_ylim()
            return np.log10(l), np.log10(r)
        return self.figure.get_ylim()

    def get_x_name(self) -> str:
        return self.figure.get_xlabel()

    def get_y_name(self) -> str:
        return self.figure.get_ylabel()

    def get_x_ticks(self) -> List[CategoricalChoiceType]:
        return list(map(lambda t: t.get_text(), self.figure.get_xticklabels()))

    def get_y_ticks(self) -> List[CategoricalChoiceType]:
        return list(map(lambda t: t.get_text(), self.figure.get_yticklabels()))

    def get_target_name(self) -> Optional[str]:
        target_name = self.figure.figure.axes[-1].get_ylabel()
        if target_name == "":
            return None
        return target_name

    def get_x_is_log(self) -> bool:
        return self.figure.get_xscale() == "log"

    def get_y_is_log(self) -> bool:
        return self.figure.get_yscale() == "log"
