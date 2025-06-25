"""
Spark with Ask-and-Tell
========================

An example showing how to use Optuna's ask-and-tell interface with Apache Spark
to distribute the evaluation of trials.

This script uses PySpark's RDD to parallelize a simple quadratic function.
"""

import optuna

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None  # Allow docs to build without pyspark


def evaluate(x):
    # Simulates a trial evaluation on a Spark worker node
    return (x - 2) ** 2


if __name__ == "__main__":
    if SparkSession is None:
        print("This example requires pyspark. Please install it to run this script.")
    else:
        spark = SparkSession.builder.appName("OptunaSparkExample").getOrCreate()
        study = optuna.create_study()

        for i in range(20):
            trial = study.ask()
            x = trial.suggest_float("x", -10, 10)

            # Simulate distributed evaluation with Spark
            rdd = spark.sparkContext.parallelize([x])
            result = rdd.map(evaluate).collect()[0]

            study.tell(trial, result)
            print(f"Trial {i + 1}: x = {x:.4f}, result = {result:.4f}")

        print("Best trial:", study.best_trial)
        spark.stop()
