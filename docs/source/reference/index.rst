API Reference
=============


How to read the API documentation?
----------------------------------

The API reference style is based on `Google Style Python Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
For more information, please refer to the `Coding Style Conventions <https://github.com/optuna/optuna/wiki/Coding-Style-Conventions#docstrings>`_.
A typical example of our docstring looks like following:

.. code-block:: python

    def some_function(self, number: int, ...) -> int:
        """Description of a method.

        More detailed explanation for the API...
        ...

        Example:

            .. testcode::

                your_awesome_number = 1
                some_function(your_awesome_number)

                ...

        Args:
            number:
                Some description for the argument here.
            ...

        Raises:
            TypeError:
                Describe here when an exception is raised.
            ...


Each documentation shows information about `Parameters` (Args), `Raises` and `Return type`.
Types of parameters and return value are based on type hints, which are annotated to each methods in our codebase.
It also contains `Methods` and `Attributes` in the class documentation and some API shows a code example for you.
Although we enumerate exceptions that an API would raise in our documentation on best effort,
it could raise other exception in runtime because an exception could be raised in third party libraries.
An undocumented exception is not a specification.
If you find a documentation doesn't show an exception that we should enumerate, we'd love to review your PR!


Modules
-------

.. toctree::
    :maxdepth: 1

    optuna
    cli
    distributions
    exceptions
    importance
    integration
    logging
    multi_objective/index
    pruners
    samplers
    storages
    study
    trial
    visualization/index
