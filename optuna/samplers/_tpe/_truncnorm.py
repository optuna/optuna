# This file contains the codes from SciPy project.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import sys
from typing import Callable

import numpy as np


_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def _log_sum(log_p: float, log_q: float) -> float:
    if log_p > log_q:
        log_p, log_q = log_q, log_p
    return math.log1p(math.exp(log_p - log_q)) + log_q


def _log_diff(log_p: float, log_q: float) -> float:
    # returns log(q - p).
    # assuming that log_q is always greater than log_q
    return math.log1p(-math.exp(log_q - log_p)) + log_p


def _ndtr(a: float) -> float:
    x = a / 2**0.5
    z = abs(x)

    if z < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 0.5 * math.erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


def _log_ndtr(a: float) -> float:
    if a > 6:
        return -_ndtr(-a)
    if a > -20:
        return math.log(_ndtr(a))

    log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
    denom_cons = 1 / a**2
    sign = 1
    i = 0

    while abs(last_total - right_hand_side) > sys.float_info.epsilon:
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        right_hand_side += sign * numerator * denom_factor

    return log_LHS + math.log(right_hand_side)


def _norm_logpdf(x: float) -> float:
    return -(x**2) / 2.0 - _norm_pdf_logC


def _log_gauss_mass(a: float, b: float) -> float:
    """Log of Gaussian probability mass within an interval"""

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail

    def mass_case_left(a: float, b: float) -> float:
        return _log_diff(_log_ndtr(b), _log_ndtr(a))

    def mass_case_right(a: float, b: float) -> float:
        return mass_case_left(-b, -a)

    def mass_case_central(a: float, b: float) -> float:
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return math.log1p(-_ndtr(a) - _ndtr(-b))

    if b <= 0:
        return mass_case_left(a, b)
    elif a > 0:
        return mass_case_right(a, b)
    else:
        return mass_case_central(a, b)


def _bisect(f: Callable[[float], float], a: float, b: float, c: float) -> float:
    if f(a) > c:
        a, b = b, a
    # TODO(amylase): Justify this constant
    for _ in range(100):
        m = (a + b) / 2
        if f(m) < c:
            a = m
        else:
            b = m
    return m


def _ndtri_exp(y: float) -> float:
    # TODO(amylase): Justify this constant
    return _bisect(_log_ndtr, -100, +100, y)


def ppf(q: float, a: float, b: float) -> float:
    if a == b:
        return np.nan
    if q == 0:
        return a
    if q == 1:
        return b

    def ppf_left(q: float, a: float, b: float) -> float:
        log_Phi_x = _log_sum(_log_ndtr(a), math.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)

    def ppf_right(q: float, a: float, b: float) -> float:
        log_Phi_x = _log_sum(_log_ndtr(-b), math.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)

    if a < 0:
        return ppf_left(q, a, b)
    else:
        return ppf_right(q, a, b)


def logpdf(x: float, a: float, b: float) -> float:
    if a == b:
        return np.nan
    if x < a or b < x:
        return -np.inf
    return _norm_logpdf(x) - _log_gauss_mass(a, b)
