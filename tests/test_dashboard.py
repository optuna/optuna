import tempfile

import pfnopt
import pfnopt.client


def _create_some_study():
    # type: () -> pfnopt.Study

    def f(client):
        # type: (pfnopt.client.BaseClient) -> float

        x = client.sample_uniform('x', -10, 10)
        y = client.sample_loguniform('y', 10, 20)
        z = client.sample_categorical('z', (10, 20.5, '30'))

        return x ** 2 + y ** 2 + float(z)

    return pfnopt.minimize(f, n_trials=100)


def test_write():
    # type: () -> None

    study = _create_some_study()

    with tempfile.NamedTemporaryFile('r') as tf:
        pfnopt.dashboard.write(study, tf.name)

        html = tf.read()
        assert '<body>' in html
        assert 'bokeh' in html
