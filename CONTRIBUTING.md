# Contribution Guidelines

It’s an honor to have you on board!

We are proud of this project and have been working to make it great since day one.
We believe you will love it, and we know there’s room for improvement.
We want to
- implement features that make what you want to do possible and/or easy.
- write more tutorials and [examples](https://github.com/optuna/optuna-examples) that help you get familiar with Optuna.
- make issues and pull requests on GitHub fruitful.
- have more conversations and discussions on [GitHub Discussions](https://github.com/optuna/optuna/discussions).

We need your help and everything about Optuna you have in your mind pushes this project forward.
Join Us!

If you feel like giving a hand, here are some ways:
- Implement a feature
    - If you have some cool idea, please open an issue first to discuss design to make your idea in a better shape.
- Send a patch
    - Dirty your hands by tackling [issues with `contribution-welcome` label](https://github.com/optuna/optuna/issues?q=is%3Aissue+is%3Aopen+label%3Acontribution-welcome)
- Report a bug
    - If you find a bug, please report it! Your reports are important.
- Fix/Improve documentation
    - Documentation gets outdated easily and can always be better, so feel free to fix and improve
- Let us and the Optuna community know your ideas and thoughts.
    - __Contribution to Optuna includes not only sending pull requests, but also writing down your comments on issues and pull requests by others, and joining conversations/discussions on [GitHub Discussions](https://github.com/optuna/optuna/discussions).__
    - Also, sharing how you enjoy Optuna is a huge contribution! If you write a blog, let us know about it!


## Pull Request Guidelines

If you make a pull request, please follow the guidelines below:

- [Setup Optuna](#setup-optuna)
- [Checking the Format, Coding Style, and Type Hints](#checking-the-format-coding-style-and-type-hints)
- [Documentation](#documentation)
- [Unit Tests](#unit-tests)
- [Continuous Integration and Local Verification](#continuous-integration-and-local-verification)
- [Creating a Pull Request](#creating-a-pull-request)

Detailed conventions and policies to write, test, and maintain Optuna code are described in the [Optuna Wiki](https://github.com/optuna/optuna/wiki).

- [Coding Style Conventions](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions)
- [Deprecation Policy](https://github.com/optuna/optuna/wiki/Deprecation-policy)
- [Test Policy](https://github.com/optuna/optuna/wiki/Test-Policy)

### Setup Optuna

First of all, fork Optuna on GitHub.
You can learn about fork in the official [documentation](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo).

After forking, download and install Optuna on your computer.

```bash
git clone git@github.com:YOUR_NAME/optuna.git
cd optuna
pip install -e .
```

### Checking the Format, Coding Style, and Type Hints

Code is formatted with [black](https://github.com/psf/black),
and docstrings are formatted with [blackdoc](https://github.com/keewis/blackdoc).
Coding style is checked with [flake8](http://flake8.pycqa.org) and [isort](https://pycqa.github.io/isort/),
and additional conventions are described in the [Wiki](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions).
Type hints, [PEP484](https://www.python.org/dev/peps/pep-0484/), are checked with [mypy](http://mypy-lang.org/).

You can check the format, coding style, and type hints at the same time just by executing a script `formats.sh`.
If your environment is missing some dependencies such as black, blackdoc, flake8, isort or mypy,
you will be asked to install them.
The following commands automatically fix format errors by auto-formatters.

```bash
# Install auto-formatters.
$ pip install ".[checking]"

$ ./formats.sh
```

### Documentation

When adding a new feature to the framework, you also need to document it in the reference.
The documentation source is stored under the [docs](./docs) directory and written in [reStructuredText format](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).

To build the documentation, you need to run:

```bash
pip install -e ".[document]"
```
Note that the above command might try to install PyTorch without CUDA to your environment even if your environment has CUDA version already.

Then you can build the documentation in HTML format locally:

```bash
cd docs
make html
```

HTML files are generated under `build/html` directory. Open `index.html` with the browser and see
if it is rendered as expected.

Optuna's tutorial is built with [Sphinx-Gallery](https://sphinx-gallery.github.io/stable/index.html) and
some other requirements like [LightGBM](https://github.com/microsoft/LightGBM) and [PyTorch](https://pytorch.org) meaning that
all .py files in `tutorial` directory are run during the documentation build if there's no build cache.
Whether you edit any tutorial or not doesn't matter.

To avoid having to run the tutorials, you may download executed tutorial artifacts named "tutorial" from our CI (see the capture below) and put them in `docs/build` before
extracting the files in the zip to `docs/source/tutorial` directory.
Note that the CI runs with Python 3.8 and the generated artifacts contain pickle files.
The pickle files are serialized with [the protocol version 5](https://docs.python.org/3/library/pickle.html#data-stream-format) so you will see the error with Python 3.7 or older.
Please use Python 3.8 or later if you build the documentation with artifacts.

![image](https://user-images.githubusercontent.com/16191443/107472296-0b211400-6bb2-11eb-9203-e2c42ce499ad.png)

**Writing a Tutorial**
Tutorials are part of Optuna’s documentation.
Optuna depends on Sphinx to build the documentation HTML files from the corresponding reStructuredText (`.rst`) files in the docs/source directory,
but as you may notice, [Tutorial directory](https://github.com/optuna/optuna/tree/master/tutorial) does not have any `.rst` files. Instead, it has a bunch of Python (`.py`) files.
We have [Sphinx Gallery](https://sphinx-gallery.github.io/stable/index.html) that executes those `.py` files and generates `.rst` files with standard outputs from them and corresponding Jupyter Notebook (`.ipynb`) files.
These generated `.rst` and `.ipynb` files are written to the docs/source/tutorial directory.
The output directory (docs/source/tutorial) and source (tutorial) directory are configured in [`sphinx_gallery_conf` of docs/source/conf.py](https://github.com/optuna/optuna/blob/2e14273cab87f13edeb9d804a43bd63c44703cb5/docs/source/conf.py#L189-L199). These generated `.rst` files are handled by Sphinx like the other `.rst` files. The generated `.ipynb` files are hosted on Optuna’s documentation page and downloadable (check [Optuna tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)).

The order of contents on [tutorial top page](https://optuna.readthedocs.io/en/stable/tutorial/index.html) is determined by two keys: one is the subdirectory name of tutorial and the other is the filename (note that there are some alternatives as documented in [Sphinx Gallery - sorting](https://sphinx-gallery.github.io/stable/gen_modules/sphinx_gallery.sorting.html?highlight=filenamesortkey), but we chose this key in https://github.com/optuna/optuna/blob/2e14273cab87f13edeb9d804a43bd63c44703cb5/docs/source/conf.py#L196).
Optuna’s tutorial directory has two directories: (1) [10_key_features](https://github.com/optuna/optuna/tree/master/tutorial/10_key_features), which is meant to be aligned with and explain the key features listed on [README.md](https://github.com/optuna/optuna#key-features) and (2) [20_recipes](https://github.com/optuna/optuna/tree/master/tutorial/20_recipes), whose contents showcase how to use Optuna features conveniently.
When adding new content to the Optuna tutorials, place it in `20_recipes` and its file name should conform to the other names, for example, `777_cool_feature.py`.
In general, please number the prefix for your file consecutively with the last number. However, this is not mandatory and if you think your content deserves the smaller number (the order of recipes does not have a specific meaning, but in general, order could convey the priority order to readers), feel free to propose the renumbering in your PR.

You may want to refer to the Sphinx Gallery for the syntax of `.py` files processed by Sphinx Gallery.
Two specific conventions and limitations for Optuna tutorials:
1. 99 #s for block separation as in https://github.com/optuna/optuna/blob/2e14273cab87f13edeb9d804a43bd63c44703cb5/tutorial/10_key_features/001_first.py#L19
2. Execution time of the new content needs to be less than three minutes. This limitation derives from Read The Docs. If your content runs some hyperparameter optimization, set the `timeout` to 180 or less. You can check this limitation on [Read the Docs - Build Process](https://docs.readthedocs.io/en/stable/builds.html).


### Unit Tests

When adding a new feature or fixing a bug, you also need to write sufficient test code.
We use [pytest](https://pytest.org/) as the testing framework and
unit tests are stored under the [tests directory](./tests).

Please install some required packages at first.
```bash
# Install required packages to test all modules without visualization and integration modules.
pip install ".[test]"

# Install required packages to test all modules including visualization and integration modules.
pip install ".[optional,integration]" -f https://download.pytorch.org/whl/torch_stable.html
```

You can run your tests as follows:

```bash
# Run all the unit tests.
pytest

# Run all the unit tests defined in the specified test file.
pytest tests/${TARGET_TEST_FILE_NAME}

# Run the unit test function with the specified name defined in the specified test file.
pytest tests/${TARGET_TEST_FILE_NAME} -k ${TARGET_TEST_FUNCTION_NAME}
```

See also the [Optuna Test Policy](https://github.com/optuna/optuna/wiki/Test-Policy), which describes the principles to write and maintain Optuna tests to meet certain quality requirements.

### Continuous Integration and Local Verification

Optuna repository uses GitHub Actions.

### Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

First, the **title** of your pull request should:

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

*The title will be directly visible in the release notes.*

For example:

- Introduces Tree-structured Parzen Estimator to `optuna.samplers`

Second, the **description** of your pull request should:

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks

## Learning Optuna's Implementation

With Optuna actively being developed and the amount of code growing,
it has become difficult to get a hold of the overall flow from reading the code.
So we created a tiny program called [Minituna](https://github.com/CyberAgentAILab/minituna).
Once you get a good understanding of how Minituna is designed, it will not be too difficult to read the Optuna code.
We encourage you to practice reading the Minituna code with the following article.

[An Introduction to the Implementation of Optuna, a Hyperparameter Optimization Framework](https://medium.com/optuna/an-introduction-to-the-implementation-of-optuna-a-hyperparameter-optimization-framework-33995d9ec354)

