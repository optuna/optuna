import argparse
import os
from typing import Callable
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
import warnings

from jinja2 import Environment
from jinja2 import FileSystemLoader
import matplotlib.pylab as plt
import optuna
from optuna import Study
from optuna.exceptions import ExperimentalWarning
import optuna.visualization as plotly_visualization
import optuna.visualization.matplotlib as matplotlib_visualization

from studies import create_intermediate_value_studies
from studies import create_multi_objective_studies
from studies import create_pytorch_study
from studies import create_single_objective_studies


try:
    from optuna_fast_fanova import FanovaImportanceEvaluator
except ImportError:
    from optuna.importance import FanovaImportanceEvaluator

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import plotly.graph_objs as go

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", help="output directory (default: %(default)s)", default="tmp")
parser.add_argument("--width", help="plot width (default: %(default)s)", type=int, default=800)
parser.add_argument("--height", help="plot height (default: %(default)s)", type=int, default=600)
parser.add_argument("--heavy", help="create studies that takes long time", action="store_true")
args = parser.parse_args()

template_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")]
abs_output_dir = os.path.abspath(args.output_dir)
dpi = 100
plt.rcParams["figure.figsize"] = (args.width / dpi, args.height / dpi)


def generate_plot_files(
    studies: List[Study],
    base_dir: str,
    plotly_plot: Callable[[Study], "go.Figure"],
    matplotlib_plot: Callable[[Study], "Axes"],
    filename_prefix: str,
) -> List[Tuple[Study, str, str]]:
    files = []
    for study in studies:
        plotly_filename = f"{filename_prefix}-{study.study_name}-plotly.png"
        plotly_filepath = os.path.join(base_dir, plotly_filename)
        matplotlib_filename = f"{filename_prefix}-{study.study_name}-matplotlib.png"
        matplotlib_filepath = os.path.join(base_dir, matplotlib_filename)
        try:
            plotly_fig = plotly_plot(study)
            plotly_fig.update_layout(
                width=args.width,
                height=args.height,
                margin={"l": 10, "r": 10},
            )
            plotly_fig.write_image(plotly_filepath)
        except:  # NOQA
            plotly_filename = ""

        try:
            matplotlib_plot(study)
            plt.savefig(matplotlib_filepath, bbox_inches="tight", dpi=dpi)
        except:  # NOQA
            matplotlib_filename = ""

        files.append((study, plotly_filename, matplotlib_filename))
    return files


def generate_optimization_history_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_optimization_history,
        matplotlib_visualization.plot_optimization_history,
        filename_prefix="history",
    )


def generate_contour_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_contour,
        matplotlib_visualization.plot_contour,
        filename_prefix="contour",
    )


def generate_edf_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_edf,
        matplotlib_visualization.plot_edf,
        filename_prefix="edf",
    )


def generate_slice_plots(studies: List[Study], base_dir: str) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_slice,
        matplotlib_visualization.plot_slice,
        filename_prefix="slice",
    )


def generate_param_importances_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    seed = 0
    return generate_plot_files(
        studies,
        base_dir,
        lambda s: plotly_visualization.plot_param_importances(
            s, evaluator=FanovaImportanceEvaluator(seed=seed)
        ),
        lambda s: matplotlib_visualization.plot_param_importances(
            s, evaluator=FanovaImportanceEvaluator(seed=seed)
        ),
        filename_prefix="importance",
    )


def generate_parallel_coordinate_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_parallel_coordinate,
        matplotlib_visualization.plot_parallel_coordinate,
        filename_prefix="parcoords",
    )


def generate_intermediate_value_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_intermediate_values,
        matplotlib_visualization.plot_intermediate_values,
        filename_prefix="intermediate",
    )


def generate_pareto_front_plots(
    studies: List[Study], base_dir: str
) -> List[Tuple[Study, str, str]]:
    return generate_plot_files(
        studies,
        base_dir,
        plotly_visualization.plot_pareto_front,
        matplotlib_visualization.plot_pareto_front,
        filename_prefix="pareto-front",
    )


def main() -> None:
    if not os.path.exists(abs_output_dir):
        os.mkdir(abs_output_dir)

    env = Environment(loader=FileSystemLoader(template_dirs))
    plot_results_template = env.get_template("plot_results.html")
    list_pages_template = env.get_template("list_pages.html")

    print("Creating single objective studies")
    single_objective_studies = create_single_objective_studies()
    print("Creating multi objective studies")
    multi_objective_studies = create_multi_objective_studies()
    print("Creating studies that have intermediate values")
    intermediate_value_studies = create_intermediate_value_studies()

    if args.heavy:
        print("Creating pytorch study")
        pytorch_study = create_pytorch_study()
        single_objective_studies.append(pytorch_study)
        intermediate_value_studies.insert(0, pytorch_study)

    pages: List[Tuple[str, str]] = []
    for funcname, studies, generate in [
        (
            "plot_optimization_history",
            single_objective_studies,
            generate_optimization_history_plots,
        ),
        ("plot_slice", single_objective_studies, generate_slice_plots),
        ("plot_contour", single_objective_studies, generate_contour_plots),
        ("plot_parallel_coordinate", single_objective_studies, generate_parallel_coordinate_plots),
        (
            "plot_intermediate_values",
            intermediate_value_studies,
            generate_intermediate_value_plots,
        ),
        ("plot_pareto_front", multi_objective_studies, generate_pareto_front_plots),
        ("plot_param_importances", single_objective_studies, generate_param_importances_plots),
        ("plot_edf", single_objective_studies, generate_edf_plots),
    ]:
        plot_files = generate(studies, abs_output_dir)

        with open(os.path.join(abs_output_dir, f"{funcname}.html"), "w") as f:
            f.write(plot_results_template.render(funcname=f"{funcname}()", plot_files=plot_files))

        pages.append((f"{funcname}()", f"{funcname}.html"))

    with open(os.path.join(abs_output_dir, "index.html"), "w") as f:
        f.write(list_pages_template.render(pages=pages))

    print("Generated to:", os.path.join(abs_output_dir, "index.html"))


if __name__ == "__main__":
    main()
