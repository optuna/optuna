from optuna.integration.lightgbm_autotune.alias import handling_alias_parameters


def test_handling_alias_parameters():
    params = {'reg_alpha': 0.1}
    handling_alias_parameters(params)
    assert 'reg_alpha' not in params
    assert 'lambda_l1' in params


def test_handling_alias_parameter_with_default_value():
    params = {
        'num_boost_round': 5,
        'early_stopping_rounds': 2,
        'eta': 0.5,
    }
    handling_alias_parameters(params)

    assert 'eta' not in params
    assert 'learning_rate' in params
    assert params['learning_rate'] == 0.1


def test_handling_alias_parameter():
    params = {
        'num_boost_round': 5,
        'early_stopping_rounds': 2,
        'min_data': 0.2,
    }
    handling_alias_parameters(params)
    assert 'min_data' not in params
    assert 'min_data_in_leaf' in params
    assert params['min_data_in_leaf'] == 0.2
