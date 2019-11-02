from optuna.integration.lightgbm_tuner.alias import _handling_alias_parameters


def test__handling_alias_parameters():
    # type: () -> None

    params = {'reg_alpha': 0.1}
    _handling_alias_parameters(params)
    assert 'reg_alpha' not in params
    assert 'lambda_l1' in params


def test_handling_alias_parameter_with_user_supplied_param():
    # type: () -> None

    params = {
        'num_boost_round': 5,
        'early_stopping_rounds': 2,
        'eta': 0.5,
    }
    _handling_alias_parameters(params)

    assert 'eta' not in params
    assert 'learning_rate' in params
    assert params['learning_rate'] == 0.5


def test_handling_alias_parameter():
    # type: () -> None

    params = {
        'num_boost_round': 5,
        'early_stopping_rounds': 2,
        'min_data': 0.2,
    }
    _handling_alias_parameters(params)
    assert 'min_data' not in params
    assert 'min_data_in_leaf' in params
    assert params['min_data_in_leaf'] == 0.2
