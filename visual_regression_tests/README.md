# Visual Regression Testing

## Background

It is difficult to ensure the behavior of the visualization module with unit tests, which can easily lead to test case omissions and increased complexity.
This tool prepares various studies and executes each function in Optuna's visualization module, and continuously runs on CircleCI.
The output files (HTML and png images) are uploaded to CircleCI Artifacts, so that we can easily check those images to uncover regression bugs.

## How to Run locally

Please install dependencies and then execute visual_regression_tests.py like the following.

```python
$ pip install -e ".[visual-regression-test]"
$ cd visual_regression_tests/
$ python visual_regression_tests.py --output-dir /tmp/foo --heavy
Generated to: /tmp/foo/index.html
```

Then you can check output images by opening "/tmp/foo/index.html" on your Web browser.
