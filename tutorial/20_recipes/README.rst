.. _recipes:

Recipes
-------

Showcases the recipes that might help you using Optuna with comfort.

Spark with Ask-and-Tell
------------------------

This example demonstrates how to use Optuna's ask-and-tell interface with Apache Spark
to distribute the evaluation of trials. This is a minimal example to illustrate how distributed
trial evaluation can be performed in Spark. In real-world scenarios, the evaluation function
would typically involve more complex or expensive computations.

For more details, see the
`Optuna ask-and-tell documentation <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html>`_.

.. literalinclude:: 014_spark_ask_and_tell.py
   :language: python
   :linenos:
