.. _distributed:

Distributed Optimization
========================

There is no complicated setup but just sharing the same study name among nodes/processes.
For example, let's run 2 processes as follows.
They are getting parameter suggestions based on shared trials' history stored in the DB.

Process 1:

.. code-block:: bash

    $ optuna study optimize hoge.py objective --timeout=120 --storage='sqlite:///example.db' --study='<STUDY_NAME>'
    [I 2018-05-09 16:26:38,143] Finished a trial resulted in value: 9.566755107945783. Current best value is 1.3905266234395878e-07 with parameters: {'x': 1.999627102343338}.
    [I 2018-05-09 16:26:38,291] Finished a trial resulted in value: 26.186520972384656. Current best value is 1.3905266234395878e-07 with parameters: {'x': 1.999627102343338}.
    ...


Process 2 (the same command as process 1):

.. code-block:: bash

    $ optuna study optimize hoge.py objective --timeout=120 --storage='sqlite:///example.db' --study='<STUDY_NAME>'
    [I 2018-05-09 16:26:41,386] Finished a trial resulted in value: 10.7299990550174. Current best value is 1.3905266234395878e-07 with parameters: {'x': 1.999627102343338}.
    [I 2018-05-09 16:26:41,687] Finished a trial resulted in value: 1.3267230950104523. Current best value is 1.3905266234395878e-07 with parameters: {'x': 1.999627102343338}.
    ...

