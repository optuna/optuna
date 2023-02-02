import numpy as np

from optuna.visualization.matplotlib._contour import _create_zmap
from optuna.visualization.matplotlib._contour import _interpolate_zmap


def test_create_zmap() -> None:
    x_values = np.arange(10)
    y_values = np.arange(10)
    z_values = list(np.random.rand(10))

    # we are testing for exact placement of z_values
    # so also passing x_values and y_values as xi and yi
    zmap = _create_zmap(x_values.tolist(), y_values.tolist(), z_values, x_values, y_values)

    assert len(zmap) == len(z_values)
    for coord, value in zmap.items():
        # test if value under coordinate
        # still refers to original trial value
        xidx = coord[0]
        yidx = coord[1]
        assert xidx == yidx
        assert z_values[xidx] == value


def test_interpolate_zmap() -> None:
    contour_point_num = 2
    zmap = {(0, 0): 1.0, (1, 1): 4.0}
    expected = np.array([[1.0, 2.5], [2.5, 4.0]])

    actual = _interpolate_zmap(zmap, contour_point_num)

    assert np.allclose(expected, actual)
