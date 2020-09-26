import enum


class StudyDirection(enum.Enum):
    """Direction of a :class:`~optuna.study.Study`.

    Attributes:
        NOT_SET:
            Direction has not been set.
        MINIMIZE:
            :class:`~optuna.study.Study` minimizes the objective function.
        MAXIMIZE:
            :class:`~optuna.study.Study` maximizes the objective function.
    """

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2


def _get_study_direction(direction: str) -> StudyDirection:
    if direction == "minimize":
        return StudyDirection.MINIMIZE
    elif direction == "maximize":
        return StudyDirection.MAXIMIZE
    else:
        raise ValueError("Please set either 'minimize' or 'maximize' to direction.")
