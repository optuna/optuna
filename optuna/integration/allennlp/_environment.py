import os
from typing import Dict


def _is_encodable(value: str) -> bool:
    # https://github.com/allenai/allennlp/blob/master/allennlp/common/params.py#L77-L85
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> Dict[str, str]:
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}
