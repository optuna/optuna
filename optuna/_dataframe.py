import collections
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import optuna
from optuna._imports import try_import
from optuna.trial._state import TrialState


with try_import() as _imports:
    # `Study.trials_dataframe` is disabled if pandas is not available.
    import pandas as pd

# Required for type annotation in `Study.trials_dataframe`.
if not _imports.is_successful():
    pd = object  # type: ignore # NOQA


def _trials_dataframe(
    study: "optuna.Study", attrs: Tuple[str, ...], multi_index: bool
) -> "pd.DataFrame":
    _imports.check()

    trials = study.get_trials(deepcopy=False)

    # If no trials, return an empty dataframe.
    if not len(trials):
        return pd.DataFrame()

    if "value" in attrs and study._is_multi_objective():
        attrs = tuple("values" if attr == "value" else attr for attr in attrs)

    attrs_to_df_columns: Dict[str, str] = collections.OrderedDict()
    for attr in attrs:
        if attr.startswith("_"):
            # Python conventional underscores are omitted in the dataframe.
            df_column = attr[1:]
        else:
            df_column = attr
        attrs_to_df_columns[attr] = df_column

    # column_agg is an aggregator of column names.
    # Keys of column agg are attributes of `FrozenTrial` such as 'trial_id' and 'params'.
    # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
    column_agg: DefaultDict[str, Set] = collections.defaultdict(set)
    non_nested_attr = ""

    def _create_record_and_aggregate_column(
        trial: "optuna.trial.FrozenTrial",
    ) -> Dict[Tuple[str, str], Any]:

        record = {}
        for attr, df_column in attrs_to_df_columns.items():
            value = getattr(trial, attr)
            if isinstance(value, TrialState):
                # Convert TrialState to str and remove the common prefix.
                value = str(value).split(".")[-1]
            if isinstance(value, dict):
                for nested_attr, nested_value in value.items():
                    record[(df_column, nested_attr)] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            elif isinstance(value, list):
                # Expand trial.values.
                for nested_attr, nested_value in enumerate(value):
                    record[(df_column, nested_attr)] = nested_value
                    column_agg[attr].add((df_column, nested_attr))
            else:
                record[(df_column, non_nested_attr)] = value
                column_agg[attr].add((df_column, non_nested_attr))
        return record

    records = list([_create_record_and_aggregate_column(trial) for trial in trials])

    columns: List[Tuple[str, str]] = sum(
        (sorted(column_agg[k]) for k in attrs if k in column_agg), []
    )

    df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

    if not multi_index:
        # Flatten the `MultiIndex` columns where names are concatenated with underscores.
        # Filtering is required to omit non-nested columns avoiding unwanted trailing
        # underscores.
        df.columns = ["_".join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns]

    return df
