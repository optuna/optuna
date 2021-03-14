# Contribution Guidelines

It’s such an honor to have you on board!

We are proud of this project and have been trying to make this project great since day one.
We believe you will love it, however, we know there’s room for improvement.
We have to
- implement features that make what you want to do possible and/or easily.
- write more examples that help you get familiar with Optuna.
- make issues and pull requests on GitHub fruitful.
- have more conversations and discussions on Gitter.

We need your help heartily, everything about Optuna you have in your mind push this project forward.
Join Us!

If you feel like giving your hand to us, here are some ways
- Implement a feature
    - If you have some cool idea, please open an issue first to discuss design to make your idea in a better shape.
- Send a patch
    - Dirty your hands by tackling [issues with `contribution-welcome` label](https://github.com/optuna/optuna/issues?q=is%3Aissue+is%3Aopen+label%3Acontribution-welcome)
- Report a bug
    - If you find some bug, don't hesitate to report it! Your reports are really important!
- Fix/Improve documentation
    - Documentation gets outdated easily, and can always be better, so feel free to fix & improve
- Let us & the Optuna community know your ideas, thought
    - __Contribution to Optuna includes not only sending pull requests, but also writing down your comments on issues and pull requests by others, and joining conversations/discussions on [Gitter](https://gitter.im/optuna/optuna).__
    - Also, sharing how you enjoy Optuna is a huge contribution! If you write some blog, let us know it!

If you choose to write some code, we have some conventions as follows.

- [Guidelines](#guidelines)
- [Unit Tests](#unit-tests)
- [Continuous Integration and Local Verification](#continuous-integration-and-local-verification)
- [Creating a Pull Request](#creating-a-pull-request)

## Guidelines

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

You can check the format, coding style, and type hint at the same time just by executing a script `formats.sh`.
If your environment misses some dependencies such as black, blackdoc, flake8, isort or mypy,
you will be asked to install them.

You can also check them using [tox](https://tox.readthedocs.io/en/latest/) like below.

```
$ pip install tox
$ tox -e flake8 -e black -e blackdoc -e isort -e mypy
```

If you catch format errors, you can automatically fix them by auto-formatters.

```bash
# Install auto-formatters.
$ pip install .[checking]

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

To avoid having to run the tutorials, you may download executed tutorial artifacts nanmed "tutorial" from our CI (see the capture below) and put them in `docs/build` before
extract the files in the zip to `docs/source/tutorial` directory.

![image](https://user-images.githubusercontent.com/16191443/107472296-0b211400-6bb2-11eb-9203-e2c42ce499ad.png)

## Unit Tests

When adding a new feature or fixing a bug, you also need to write sufficient test code.
We use [pytest](https://pytest.org/) as the testing framework and
unit tests are stored under the [tests directory](./tests).

You can run all your tests as follows:

```bash
# Run all the unit tests.
pytest

# Run all the unit tests defined in the specified test file.
pytest tests/${TARGET_TEST_FILE_NAME}
```

## Continuous Integration and Local Verification

Optuna repository uses GitHub Actions and CircleCI.

Currently, we are migrating to GitHub Actions but still we use CirclCI for a test of `document`
because it makes it much easier to check built documentation.

### Local Verification

By installing [`act`](https://github.com/nektos/act#installation) and Docker, you can run
tests written for GitHub Actions locally.

```bash
JOB_NAME=checks
act -j $JOB_NAME
```

Currently, you can run the following jobs: `documentation` and `doctest` may not be executable depending on your choice of docker image of act.

- `checks`    
  - Checking the format, coding style, and type hints
- `docuemtnation`
  - Builds documentation including tutorial
- `doctest`
  - Runs doctest

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

### Title

The title of your pull request should

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

*The title will be directly visible in the release notes.*

#### Example

Introduces Tree-structured Parzen Estimator to `optuna.samplers`

### Description

The description of your pull request should

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks
