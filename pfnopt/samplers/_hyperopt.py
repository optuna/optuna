# This file is from the hyperopt project

"""
Copyright (c) 2013, James Bergstra
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of hyperopt nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Graphical model (GM)-based optimization algorithm using Theano
"""

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/jaberg/hyperopt"

import logging
import time

import numpy as np
from scipy.special import erf
from typing import Dict, Any

logger = logging.getLogger(__name__)

EPS = 1e-12

# -- default linear forgetting. don't try to change by writing this variable
# because it's captured in function default args when this file is read
DEFAULT_LF = 25


adaptive_parzen_samplers = {}  # type: Dict[Any, Any]


def adaptive_parzen_sampler(name):
    def wrapper(f):
        assert name not in adaptive_parzen_samplers
        adaptive_parzen_samplers[name] = f
        return f
    return wrapper


#
# These are some custom distributions
# that are used to represent posterior distributions.
#

# -- Categorical

def categorical_lpdf(sample, p, upper):
    """
    """
    if sample.size:
        return np.log(np.asarray(p)[sample])
    else:
        return np.asarray([])


# -- Bounded Gaussian Mixture Model (BGMM)

def GMM1(weights, mus, sigmas, low=None, high=None, q=None, rng=None,
        size=()):
    """Sample from truncated 1-D Gaussian Mixture Model"""
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    assert len(weights) == len(mus) == len(sigmas)
    n_samples = np.prod(size)
    #n_components = len(weights)
    if low is None and high is None:
        # -- draw from a standard GMM
        active = np.argmax(rng.multinomial(1, weights, (n_samples,)), axis=1)
        samples = rng.normal(loc=mus[active], scale=sigmas[active])
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError('low >= high', (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(draw)
    samples = np.reshape(np.asarray(samples), size)
    #print 'SAMPLES', samples
    if q is None:
        return samples
    else:
        return np.round(samples / q) * q


def normal_cdf(x, mu, sigma):
    top = (x - mu)
    bottom = np.maximum(np.sqrt(2) * sigma, EPS)
    z = top / bottom
    return 0.5 * (1 + erf(z))


def GMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    verbose = 0
    samples, weights, mus, sigmas = map(np.asarray,
            (samples, weights, mus, sigmas))
    if samples.size == 0:
        return np.asarray([])
    if weights.ndim != 1:
        raise TypeError('need vector of weights', weights.shape)
    if mus.ndim != 1:
        raise TypeError('need vector of mus', mus.shape)
    if sigmas.ndim != 1:
        raise TypeError('need vector of sigmas', sigmas.shape)
    assert len(weights) == len(mus) == len(sigmas)
    _samples = samples
    samples = _samples.flatten()

    if verbose:
        print('GMM1_lpdf:samples', set(samples))
        print('GMM1_lpdf:weights', weights)
        print('GMM1_lpdf:mus', mus)
        print('GMM1_lpdf:sigmas', sigmas)
        print('GMM1_lpdf:low', low)
        print('GMM1_lpdf:high', high)
        print('GMM1_lpdf:q', q)

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
                weights * (
                    normal_cdf(high, mus, sigmas)
                    - normal_cdf(low, mus, sigmas)))

    if q is None:
        dist = samples[:, None] - mus
        mahal = (dist / np.maximum(sigmas, EPS)) ** 2
        # mahal shape is (n_samples, n_components)
        Z = np.sqrt(2 * np.pi * sigmas ** 2)
        coef = weights / Z / p_accept
        rval = logsum_rows(- 0.5 * mahal + np.log(coef))
    else:
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + q / 2.0
            else:
                ubound = np.minimum(samples + q / 2.0, high)
            if low is None:
                lbound = samples - q / 2.0
            else:
                lbound = np.maximum(samples - q / 2.0, low)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * normal_cdf(ubound, mu, sigma)
            inc_amt -= w * normal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)

    if verbose:
        print('GMM1_lpdf:rval:', dict(zip(samples, rval)))

    rval.shape = _samples.shape
    return rval


# -- Mixture of Log-Normals

def lognormal_cdf(x, mu, sigma):
    # wikipedia claims cdf is
    # .5 + .5 erf( log(x) - mu / sqrt(2 sigma^2))
    #
    # the maximum is used to move negative values and 0 up to a point
    # where they do not cause nan or inf, but also don't contribute much
    # to the cdf.
    if len(x) == 0:
        return np.asarray([])
    if x.min() < 0:
        raise ValueError('negative arg to lognormal_cdf', x)
    olderr = np.seterr(divide='ignore')
    try:
        top = np.log(np.maximum(x, EPS)) - mu
        bottom = np.maximum(np.sqrt(2) * sigma, EPS)
        z = top / bottom
        return .5 + .5 * erf(z)
    finally:
        np.seterr(**olderr)


def lognormal_lpdf(x, mu, sigma):
    # formula copied from wikipedia
    # http://en.wikipedia.org/wiki/Log-normal_distribution
    assert np.all(sigma >= 0)
    sigma = np.maximum(sigma, EPS)
    Z = sigma * x * np.sqrt(2 * np.pi)
    E = 0.5 * ((np.log(x) - mu) / sigma) ** 2
    rval = -E - np.log(Z)
    return rval


def qlognormal_lpdf(x, mu, sigma, q):
    # casting rounds up to nearest step multiple.
    # so lpdf is log of integral from x-step to x+1 of P(x)

    # XXX: subtracting two numbers potentially very close together.
    return np.log(
            lognormal_cdf(x, mu, sigma)
            - lognormal_cdf(x - q, mu, sigma))


def LGMM1(weights, mus, sigmas, low=None, high=None, q=None,
        rng=None, size=()):
    weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
    n_samples = np.prod(size)
    #n_components = len(weights)
    if low is None and high is None:
        active = np.argmax(
                rng.multinomial(1, weights, (n_samples,)),
                axis=1)
        assert len(active) == n_samples
        samples = np.exp(
                rng.normal(
                    loc=mus[active],
                    scale=sigmas[active]))
    else:
        # -- draw from truncated components
        # TODO: one-sided-truncation
        low = float(low)
        high = float(high)
        if low >= high:
            raise ValueError('low >= high', (low, high))
        samples = []
        while len(samples) < n_samples:
            active = np.argmax(rng.multinomial(1, weights))
            draw = rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples.append(np.exp(draw))
        samples = np.asarray(samples)

    samples = np.reshape(np.asarray(samples), size)
    if q is not None:
        samples = np.round(samples / q) * q
    return samples


def logsum_rows(x):
    R, C = x.shape
    m = x.max(axis=1)
    return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


def LGMM1_lpdf(samples, weights, mus, sigmas, low=None, high=None, q=None):
    samples, weights, mus, sigmas = map(np.asarray,
            (samples, weights, mus, sigmas))
    assert weights.ndim == 1
    assert mus.ndim == 1
    assert sigmas.ndim == 1
    _samples = samples
    if samples.ndim != 1:
        samples = samples.flatten()

    if low is None and high is None:
        p_accept = 1
    else:
        p_accept = np.sum(
                weights * (
                    normal_cdf(high, mus, sigmas)
                    - normal_cdf(low, mus, sigmas)))

    if q is None:
        # compute the lpdf of each sample under each component
        lpdfs = lognormal_lpdf(samples[:, None], mus, sigmas)
        rval = logsum_rows(lpdfs + np.log(weights))
    else:
        # compute the lpdf of each sample under each component
        prob = np.zeros(samples.shape, dtype='float64')
        for w, mu, sigma in zip(weights, mus, sigmas):
            if high is None:
                ubound = samples + q / 2.0
            else:
                ubound = np.minimum(samples + q / 2.0, np.exp(high))
            if low is None:
                lbound = samples - q / 2.0
            else:
                lbound = np.maximum(samples - q / 2.0, np.exp(low))
            lbound = np.maximum(0, lbound)
            # -- two-stage addition is slightly more numerically accurate
            inc_amt = w * lognormal_cdf(ubound, mu, sigma)
            inc_amt -= w * lognormal_cdf(lbound, mu, sigma)
            prob += inc_amt
        rval = np.log(prob) - np.log(p_accept)
    rval.shape = _samples.shape
    return rval


#
# This is the weird heuristic ParzenWindow estimator used for continuous
# distributions in various ways.
#

def adaptive_parzen_normal_orig(mus, prior_weight, prior_mu, prior_sigma):
    """
    A heuristic estimator for the mu and sigma values of a GMM
    TODO: try to find this heuristic in the literature, and cite it - Yoshua
    mentioned the term 'elastic' I think?

    mus - matrix (N, M) of M, N-dimensional component centers
    """
    mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != 'object'

    if mus.ndim != 1:
        raise TypeError('mus must be vector', mus)
    if len(mus) == 0:
        mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
    elif len(mus) == 1:
        mus = np.asarray([prior_mu] + [mus[0]])
        sigma = np.asarray([prior_sigma, prior_sigma * .5])
    elif len(mus) >= 2:
        order = np.argsort(mus)
        mus = mus[order]
        sigma = np.zeros_like(mus)
        sigma[1:-1] = np.maximum(
                mus[1:-1] - mus[0:-2],
                mus[2:] - mus[1:-1])
        if len(mus) > 2:
            lsigma = mus[2] - mus[0]
            usigma = mus[-1] - mus[-3]
        else:
            lsigma = mus[1] - mus[0]
            usigma = mus[-1] - mus[-2]

        sigma[0] = lsigma
        sigma[-1] = usigma

        # XXX: is sorting them necessary anymore?
        # un-sort the mus and sigma
        mus[order] = mus.copy()
        sigma[order] = sigma.copy()

        if not np.all(mus_orig == mus):
            print('orig', mus_orig)
            print('mus', mus)
        assert np.all(mus_orig == mus)

        # put the prior back in
        mus = np.asarray([prior_mu] + list(mus))
        sigma = np.asarray([prior_sigma] + list(sigma))

    maxsigma = prior_sigma
    # -- magic formula:
    minsigma = prior_sigma / np.sqrt(1 + len(mus))

    #print 'maxsigma, minsigma', maxsigma, minsigma
    sigma = np.clip(sigma, minsigma, maxsigma)

    weights = np.ones(len(mus), dtype=mus.dtype)
    weights[0] = prior_weight

    #print weights.dtype
    weights = weights / weights.sum()
    if 0:
        print('WEIGHTS', weights)
        print('MUS', mus)
        print('SIGMA', sigma)

    return weights, mus, sigma


def linear_forgetting_weights(N, LF):
    assert N >= 0
    assert LF > 0
    if N == 0:
        return np.asarray([])
    elif N < LF:
        return np.ones(N)
    else:
        ramp = np.linspace(1.0 / N, 1.0, num=N - LF)
        flat = np.ones(LF)
        weights = np.concatenate([ramp, flat], axis=0)
        assert weights.shape == (N,), (weights.shape, N)
        return weights

# XXX: make TPE do a post-inference pass over the pyll graph and insert
# non-default LF argument
# @scope.define_info(o_len=3)
def adaptive_parzen_normal(mus, prior_weight, prior_mu, prior_sigma,
        LF=DEFAULT_LF):
    """
    mus - matrix (N, M) of M, N-dimensional component centers
    """
    #mus_orig = np.array(mus)
    mus = np.array(mus)
    assert str(mus.dtype) != 'object'

    if mus.ndim != 1:
        raise TypeError('mus must be vector', mus)
    if len(mus) == 0:
        srtd_mus = np.asarray([prior_mu])
        sigma = np.asarray([prior_sigma])
        prior_pos = 0
    elif len(mus) == 1:
        if prior_mu < mus[0]:
            prior_pos = 0
            srtd_mus = np.asarray([prior_mu, mus[0]])
            sigma = np.asarray([prior_sigma, prior_sigma * .5])
        else:
            prior_pos = 1
            srtd_mus = np.asarray([mus[0], prior_mu])
            sigma = np.asarray([prior_sigma * .5, prior_sigma])
    elif len(mus) >= 2:

        # create new_mus, which is sorted, and in which
        # the prior has been inserted
        order = np.argsort(mus)
        prior_pos = np.searchsorted(mus[order], prior_mu)
        srtd_mus = np.zeros(len(mus) + 1)
        srtd_mus[:prior_pos] = mus[order[:prior_pos]]
        srtd_mus[prior_pos] = prior_mu
        srtd_mus[prior_pos + 1:] = mus[order[prior_pos:]]
        sigma = np.zeros_like(srtd_mus)
        sigma[1:-1] = np.maximum(
                srtd_mus[1:-1] - srtd_mus[0:-2],
                srtd_mus[2:] - srtd_mus[1:-1])
        lsigma = srtd_mus[1] - srtd_mus[0]
        usigma = srtd_mus[-1] - srtd_mus[-2]
        sigma[0] = lsigma
        sigma[-1] = usigma

    if LF and LF < len(mus):
        unsrtd_weights = linear_forgetting_weights(len(mus), LF)
        srtd_weights = np.zeros_like(srtd_mus)
        assert len(unsrtd_weights) + 1 == len(srtd_mus)
        srtd_weights[:prior_pos] = unsrtd_weights[order[:prior_pos]]
        srtd_weights[prior_pos] = prior_weight
        srtd_weights[prior_pos + 1:] = unsrtd_weights[order[prior_pos:]]

    else:
        srtd_weights = np.ones(len(srtd_mus))
        srtd_weights[prior_pos] = prior_weight

    # -- magic formula:
    maxsigma = prior_sigma / 1.0
    minsigma = prior_sigma / min(100.0, (1.0 + len(srtd_mus)))

    #print 'maxsigma, minsigma', maxsigma, minsigma
    sigma = np.clip(sigma, minsigma, maxsigma)

    sigma[prior_pos] = prior_sigma
    assert prior_sigma > 0
    assert maxsigma > 0
    assert minsigma > 0
    assert np.all(sigma > 0), (sigma.min(), minsigma, maxsigma)


    #print weights.dtype
    srtd_weights /= srtd_weights.sum()
    if 0:
        print('WEIGHTS', srtd_weights)
        print('MUS', srtd_mus)
        print('SIGMA', sigma)

    return srtd_weights, srtd_mus, sigma

#
# Adaptive Parzen Samplers
# These produce conditional estimators for various prior distributions
#

# -- Uniform


# @adaptive_parzen_sampler('uniform')
def ap_uniform_sampler(obs, prior_weight, low, high, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = adaptive_parzen_normal(obs,
            prior_weight, prior_mu, prior_sigma)
    return GMM1(weights, mus, sigmas, low=low, high=high, q=None,
            size=size, rng=rng)


# @adaptive_parzen_sampler('quniform')
def ap_quniform_sampler(obs, prior_weight, low, high, q, size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = adaptive_parzen_normal(obs,
            prior_weight, prior_mu, prior_sigma)
    return GMM1(weights, mus, sigmas, low=low, high=high, q=q,
            size=size, rng=rng)


def broadcast_best(samples, below_llik, above_llik):
    if len(samples):
        #print 'AA2', dict(zip(samples, below_llik - above_llik))
        score = below_llik - above_llik
        if len(samples) != len(score):
            raise ValueError()
        best = np.argmax(score)

        # print('broadcast_best', samples, samples[best])

        return [samples[best]] * len(samples)
    else:
        return []


def sample_uniform(obs_below, obs_above, prior_weight, low, high, size=(), rng=None):
    # Based on `ap_uniform_sampler`

    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)

    # Below
    weights_b, mus_b, sigmas_b = adaptive_parzen_normal(obs_below, prior_weight, prior_mu, prior_sigma)
    samples_b = GMM1(
        weights_b, mus_b, sigmas_b,
        low=low, high=high, q=None, size=size, rng=rng)
    llik_b = GMM1_lpdf(
        samples_b, weights_b, mus_b, sigmas_b,
        low=low, high=high, q=None)

    # Above
    weights_a, mus_a, sigmas_a = adaptive_parzen_normal(obs_above, prior_weight, prior_mu, prior_sigma)
    llik_a = GMM1_lpdf(
        samples_b, weights_a, mus_a, sigmas_a,
        low=low, high=high, q=None)

    return broadcast_best(samples_b, llik_b, llik_a)[0]  # TODO


def sample_loguniform(obs_below, obs_above, prior_weight, low, high, size=(), rng=None):
    # Based on `ap_loguniform_sampler`
    # [exp(low), exp(high)]

    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)

    # Below
    weights_b, mus_b, sigmas_b = adaptive_parzen_normal(
        np.log(obs_below), prior_weight, prior_mu, prior_sigma)
    samples_b = LGMM1(
        weights_b, mus_b, sigmas_b, low=low, high=high,
        size=size, rng=rng)
    llik_b = LGMM1_lpdf(
        samples_b, weights_b, mus_b, sigmas_b,
        low=low, high=high, q=None)

    # Above
    weights_a, mus_a, sigmas_a = adaptive_parzen_normal(
        np.log(obs_above), prior_weight, prior_mu, prior_sigma)
    llik_a = LGMM1_lpdf(
        samples_b, weights_a, mus_a, sigmas_a,
        low=low, high=high, q=None)

    return broadcast_best(samples_b, llik_b, llik_a)[0]  # TODO


def sample_quniform(obs_below, obs_above, prior_weight, low, high, size=(), rng=None, q=None):
    # Based on `ap_quniform_sampler`

    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)

    # Below
    weights_b, mus_b, sigmas_b = adaptive_parzen_normal(
        obs_below, prior_weight, prior_mu, prior_sigma)
    samples_b = GMM1(
        weights_b, mus_b, sigmas_b,
        low=low, high=high, q=q, size=size, rng=rng)
    llik_b = GMM1_lpdf(
        samples_b, weights_b, mus_b, sigmas_b,
        low=low, high=high, q=q)

    # Above
    weights_a, mus_a, sigmas_a = adaptive_parzen_normal(
        obs_above, prior_weight, prior_mu, prior_sigma)
    llik_a = GMM1_lpdf(
        samples_b, weights_a, mus_a, sigmas_a,
        low=low, high=high, q=q)

    return broadcast_best(samples_b, llik_b, llik_a)[0]  # TODO


def categorical(p, upper=None, rng=None, size=()):
    # From pyll/stochastic.py

    """Draws i with probability p[i]"""
    if len(p) == 1 and isinstance(p[0], np.ndarray):
        p = p[0]
    p = np.asarray(p)

    if size == ():
        size = (1,)
    elif isinstance(size, (int, np.number)):
        size = (size,)
    else:
        size = tuple(size)

    if size == (0,):
        return np.asarray([])
    assert len(size)

    if p.ndim == 0:
        raise NotImplementedError()
    elif p.ndim == 1:
        n_draws = int(np.prod(size))
        sample = rng.multinomial(n=1, pvals=p, size=int(n_draws))
        assert sample.shape == size + (len(p),)
        rval = np.dot(sample, np.arange(len(p)))
        rval.shape = size
        return rval
    elif p.ndim == 2:
        n_draws_, n_choices = p.shape
        n_draws, = size
        assert n_draws == n_draws_
        rval = [np.where(rng.multinomial(pvals=p[ii], n=1))[0][0]
                for ii in range(n_draws)]
        rval = np.asarray(rval)
        rval.shape = size
        return rval
    else:
        raise NotImplementedError()


def sample_categorical(obs_below, obs_above, prior_weight, upper, size=(), rng=None, LF=DEFAULT_LF):
    # Based on `ap_categorical_sampler`

    # Below
    weights_b = linear_forgetting_weights(len(obs_below), LF=LF)
    counts_b = np.bincount(obs_below, minlength=upper, weights=weights_b)
    pseudocounts_b = counts_b + prior_weight
    pseudocounts_b /= pseudocounts_b.sum()
    samples_b = categorical(pseudocounts_b, upper=upper, size=size, rng=rng)
    llik_b = categorical_lpdf(samples_b, pseudocounts_b, upper=upper)

    # Above
    weights_a = linear_forgetting_weights(len(obs_above), LF=LF)
    counts_a = np.bincount(obs_above, minlength=upper, weights=weights_a)
    pseudocounts_a = counts_a + prior_weight
    pseudocounts_a /= pseudocounts_a.sum()
    llik_a = categorical_lpdf(samples_b, pseudocounts_a, upper=upper)

    return broadcast_best(samples_b, llik_b, llik_a)[0]








"""
@adaptive_parzen_sampler('loguniform')
def ap_loguniform_sampler(obs, prior_weight, low, high,
        size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = adaptive_parzen_normal(
            scope.log(obs), prior_weight, prior_mu, prior_sigma)
    rval = scope.LGMM1(weights, mus, sigmas, low=low, high=high,
            size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qloguniform')
def ap_qloguniform_sampler(obs, prior_weight, low, high, q,
        size=(), rng=None):
    prior_mu = 0.5 * (high + low)
    prior_sigma = 1.0 * (high - low)
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(
                # -- map observations that were quantized to be below exp(low)
                #    (particularly 0) back up to exp(low) where they will
                #    interact in a reasonable way with the AdaptiveParzen
                #    thing.
                scope.maximum(
                    obs,
                    scope.maximum(  # -- protect against exp(low) underflow
                        EPS,
                        scope.exp(low)))),
            prior_weight, prior_mu, prior_sigma)
    return scope.LGMM1(weights, mus, sigmas, low, high, q=q,
            size=size, rng=rng)


# -- Normal

@adaptive_parzen_sampler('normal')
def ap_normal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, size=size, rng=rng)


@adaptive_parzen_sampler('qnormal')
def ap_qnormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            obs, prior_weight, mu, sigma)
    return scope.GMM1(weights, mus, sigmas, q=q, size=size, rng=rng)


@adaptive_parzen_sampler('lognormal')
def ap_loglognormal_sampler(obs, prior_weight, mu, sigma, size=(), rng=None):
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            scope.log(obs), prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, size=size, rng=rng)
    return rval


@adaptive_parzen_sampler('qlognormal')
def ap_qlognormal_sampler(obs, prior_weight, mu, sigma, q, size=(), rng=None):
    log_obs = scope.log(scope.maximum(obs, EPS))
    weights, mus, sigmas = scope.adaptive_parzen_normal(
            log_obs, prior_weight, mu, sigma)
    rval = scope.LGMM1(weights, mus, sigmas, q=q, size=size, rng=rng)
    return rval


# -- Categorical

@adaptive_parzen_sampler('randint')
def ap_categorical_sampler(obs, prior_weight, upper,
        size=(), rng=None, LF=DEFAULT_LF):
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    counts = scope.bincount(obs, minlength=upper, weights=weights)
    # -- add in some prior pseudocounts
    pseudocounts = counts + prior_weight
    return scope.categorical(pseudocounts / scope.sum(pseudocounts),
            upper=upper, size=size, rng=rng)


# @adaptive_parzen_sampler('categorical')
# def ap_categorical_sampler(obs, prior_weight, p, upper, size=(), rng=None,
#                            LF=DEFAULT_LF):
#     return scope.categorical(p, upper, size=size, rng
#                              =rng)

@scope.define
def tpe_cat_pseudocounts(counts, upper, prior_weight, p, size):
    #print counts
    if size == 0 or np.prod(size) == 0:
        return []
    if p.ndim == 2:
        assert np.all(p == p[0])
        p = p[0]
    pseudocounts = counts + upper * (prior_weight * p)
    return pseudocounts / np.sum(pseudocounts)

@adaptive_parzen_sampler('categorical')
def ap_categorical_sampler(obs, prior_weight, p, upper=None,
        size=(), rng=None, LF=DEFAULT_LF):
    weights = scope.linear_forgetting_weights(scope.len(obs), LF=LF)
    counts = scope.bincount(obs, minlength=upper, weights=weights)
    pseudocounts = scope.tpe_cat_pseudocounts(counts, upper, prior_weight, p, size)
    return scope.categorical(pseudocounts, upper=upper, size=size, rng=rng)
"""

#
# Posterior clone performs symbolic inference on the pyll graph of priors.
#

# @scope.define_info(o_len=2)
def ap_filter_trials(o_idxs, o_vals, l_idxs, l_vals, gamma,
        gamma_cap=DEFAULT_LF):
    """Return the elements of o_vals that correspond to trials whose losses
    were above gamma, or below gamma.
    """
    o_idxs, o_vals, l_idxs, l_vals = map(np.asarray, [o_idxs, o_vals, l_idxs,
        l_vals])

    # XXX if this is working, refactor this sort for efficiency

    # Splitting is done this way to cope with duplicate loss values.
    n_below = min(int(np.ceil(gamma * np.sqrt(len(l_vals)))), gamma_cap)
    l_order = np.argsort(l_vals)


    keep_idxs = set(l_idxs[l_order[:n_below]])
    below = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    keep_idxs = set(l_idxs[l_order[n_below:]])
    above = [v for i, v in zip(o_idxs, o_vals) if i in keep_idxs]

    #print 'AA0', below
    #print 'AA1', above

    '''
    print('ap_filter_trials')
    print(o_idxs, o_vals)
    print(l_idxs, l_vals)
    print(below, above)
    '''

    return np.asarray(below), np.asarray(above)



default_prior_weight = 1.0

# -- suggest best of this many draws on every iteration
default_n_ei_candidates = 24

# -- gamma * sqrt(n_trials) is fraction of to use as good
default_gamma = 0.25

default_n_startup_trials = 10  # 20 woeifjapowejifopawjfepowaejfpoawiejfpowaijefpaowiejfpaowefj

_default_linear_forgetting = DEFAULT_LF