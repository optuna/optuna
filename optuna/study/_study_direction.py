import enum


class StudyDirection(enum.IntEnum):
    """Direction of a :class:`~optuna.study.Study`.

    Attributes:
        MINIMIZE:
            :class:`~optuna.study.Study` minimizes the objective function.
        MAXIMIZE:
            :class:`~optuna.study.Study` maximizes the objective function.
    """

    MINIMIZE = 1
    MAXIMIZE = 2
