# This code is the modified version of erf function in FreeBSD's standard C library.
# origin: FreeBSD /usr/src/lib/msun/src/s_erf.c
# https://github.com/freebsd/freebsd-src/blob/main/lib/msun/src/s_erf.c

# /* @(#)s_erf.c 5.1 93/09/24 */
# /*
#  * ====================================================
#  * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#  *
#  * Developed at SunPro, a Sun Microsystems, Inc. business.
#  * Permission to use, copy, modify, and distribute this
#  * software is freely granted, provided that this notice
#  * is preserved.
#  * ====================================================
#  */

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import Polynomial


if TYPE_CHECKING:
    from collections.abc import Callable


erx = 8.45062911510467529297e-01
# /*
#  * In the domain [0, 2**-28], only the first term in the power series
#  * expansion of erf(x) is used.  The magnitude of the first neglected
#  * terms is less than 2**-84.
#  */
efx = 1.28379167095512586316e-01

# Coefficients for approximation to erf on [0,0.84375]

pp0 = 1.28379167095512558561e-01
pp1 = -3.25042107247001499370e-01
pp2 = -2.84817495755985104766e-02
pp3 = -5.77027029648944159157e-03
pp4 = -2.37630166566501626084e-05
pp = Polynomial([pp0, pp1, pp2, pp3, pp4])
qq1 = 3.97917223959155352819e-01
qq2 = 6.50222499887672944485e-02
qq3 = 5.08130628187576562776e-03
qq4 = 1.32494738004321644526e-04
qq5 = -3.96022827877536812320e-06
qq = Polynomial([1, qq1, qq2, qq3, qq4, qq5])

# Coefficients for approximation to erf in [0.84375,1.25]

pa0 = -2.36211856075265944077e-03
pa1 = 4.14856118683748331666e-01
pa2 = -3.72207876035701323847e-01
pa3 = 3.18346619901161753674e-01
pa4 = -1.10894694282396677476e-01
pa5 = 3.54783043256182359371e-02
pa6 = -2.16637559486879084300e-03
pa = Polynomial([pa0, pa1, pa2, pa3, pa4, pa5, pa6])
qa1 = 1.06420880400844228286e-01
qa2 = 5.40397917702171048937e-01
qa3 = 7.18286544141962662868e-02
qa4 = 1.26171219808761642112e-01
qa5 = 1.36370839120290507362e-02
qa6 = 1.19844998467991074170e-02
qa = Polynomial([1, qa1, qa2, qa3, qa4, qa5, qa6])

# Coefficients for approximation to erfc in [1.25,1/0.35]

ra0 = -9.86494403484714822705e-03
ra1 = -6.93858572707181764372e-01
ra2 = -1.05586262253232909814e01
ra3 = -6.23753324503260060396e01
ra4 = -1.62396669462573470355e02
ra5 = -1.84605092906711035994e02
ra6 = -8.12874355063065934246e01
ra7 = -9.81432934416914548592e00
ra = Polynomial([ra0, ra1, ra2, ra3, ra4, ra5, ra6, ra7])
sa1 = 1.96512716674392571292e01
sa2 = 1.37657754143519042600e02
sa3 = 4.34565877475229228821e02
sa4 = 6.45387271733267880336e02
sa5 = 4.29008140027567833386e02
sa6 = 1.08635005541779435134e02
sa7 = 6.57024977031928170135e00
sa8 = -6.04244152148580987438e-02
sa = Polynomial([1, sa1, sa2, sa3, sa4, sa5, sa6, sa7, sa8])

# Coefficients for approximation to erfc in [1/.35,28]

rb0 = -9.86494292470009928597e-03
rb1 = -7.99283237680523006574e-01
rb2 = -1.77579549177547519889e01
rb3 = -1.60636384855821916062e02
rb4 = -6.37566443368389627722e02
rb5 = -1.02509513161107724954e03
rb6 = -4.83519191608651397019e02
rb = Polynomial([rb0, rb1, rb2, rb3, rb4, rb5, rb6])
sb1 = 3.03380607434824582924e01
sb2 = 3.25792512996573918826e02
sb3 = 1.53672958608443695994e03
sb4 = 3.19985821950859553908e03
sb5 = 2.55305040643316442583e03
sb6 = 4.74528541206955367215e02
sb7 = -2.24409524465858183362e01
sb = Polynomial([1, sb1, sb2, sb3, sb4, sb5, sb6, sb7])


def _erf_right_non_big(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 1, "Input must be a 1D array."
    # NOTE(nabenabe): Add [6] to the list and use out = np.ones_like(x) to handle the big case.
    bin_inds = np.count_nonzero(x >= [[2**-28], [0.84375], [1.25], [1 / 0.35]], axis=0)
    out = np.empty_like(x)
    erf_approx_in_each_bin: list[Callable[[np.ndarray], np.ndarray]] = [
        lambda x: (1 + efx) * x,  # Tiny: x < 2**-28.
        lambda x: x * (1 + pp(z := x * x) / qq(z)),  # Small1: 2**-28 <= x < 0.84375.
        lambda x: erx + pa(s := x - 1) / qa(s),  # Small2: 0.84375 <= x < 1.25.
        # Med1: 1.25 <= x < 1 / 0.35, Med2: 1 / 0.35 <= x < 6.
        # Omit SET_LOW_WORD due to its unavailablility in NumPy and no need for high accuracy.
        lambda x: 1 - np.exp(-(z := x * x) - 0.5625 + ra(s := 1 / z) / sa(s)) / x,
        lambda x: 1 - np.exp(-(z := x * x) - 0.5625 + rb(s := 1 / z) / sb(s)) / x,
    ]
    for bin_idx, erf_approx_in_bin in enumerate(erf_approx_in_each_bin):
        if (target_inds := np.nonzero(bin_inds == bin_idx)[0]).size:
            out[target_inds] = erf_approx_in_bin(x[target_inds])

    return out


def erf(x: np.ndarray) -> np.ndarray:
    if x.size < 2000:
        return np.asarray([math.erf(v) for v in x.ravel()]).reshape(x.shape)

    a = np.abs(x).ravel()
    is_not_nan = ~np.isnan(a)
    out = np.where(is_not_nan, 1.0, np.nan)
    non_big_inds = np.nonzero(is_not_nan & (a < 6))[0]
    out[non_big_inds] = _erf_right_non_big(a[non_big_inds])
    return np.sign(x) * out.reshape(x.shape)
