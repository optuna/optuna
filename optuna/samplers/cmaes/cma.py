import math
import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class _CMA:
    def __init__(
            self,
            mean: np.ndarray,
            sigma: float,
            bounds: Optional[np.ndarray] = None,
            n_max_resampling: int = 100,
            seed: Optional[int] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        popsize = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        mu = popsize // 2

        # (eq.49)
        weights_prime = np.array(
            [math.log((popsize + 1) / 2) - math.log(i + 1) for i in range(popsize)]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

        # learning rate for the rank-one update
        # (p.17): c1 = 2 / (dim ** 2)
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        # (p.12) cmu = min(1, mu_eff / (dim ** 2))
        cmu = min(
            1 - c1,
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = popsize
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean
        self._C = np.eye(n_dim)
        self._sigma = sigma
        self._D = None  # type: Optional[np.ndarray]
        self._B = None  # type: Optional[np.ndarray]

        # bounds contains low and high of each parameter.
        self._bounds = bounds  # (n_dim, 2)-dim matrix
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

    def __getstate__(self) -> Dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            if name == "_C":
                sym1d = _compress_symmetric(self._C)
                attrs["_c_1d"] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    @property
    def population_size(self) -> int:
        return self._popsize

    @property
    def generation(self) -> int:
        return self._g

    def ask(self) -> np.ndarray:
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _sample_solution(self) -> np.ndarray:
        if self._B is None or self._D is None:
            self._C = (self._C + self._C.T) / 2
            D2, B = np.linalg.eigh(self._C)
            D = np.sqrt(D2)
            self._B, self._D = B, D

        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        y = self._B.dot(np.diag(self._D)).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return np.all(param >= self._bounds[:, 0]) and np.all(
            param <= self._bounds[:, 1]
        )

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        if len(solutions) != self._popsize:
            raise ValueError("Must tell popsize-length solutions.")

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Sample new population of search_points, for k=1, ..., popsize
        if self._B is None or self._D is None:
            self._C = (self._C + self._C.T) / 2
            D2, B = np.linalg.eigh(self._C)  # eigen decomposition for symmetric matrix.
            D = np.sqrt(D2)
        else:
            B, D = self._B, self._D
        self._B, self._D = None, None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = B.dot(np.diag(1 / D)).dot(B)  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )
        # Avoid eigendecomposition error by arithmetic overflow
        self._C += 1e-16


def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = np.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start: start + n - i] = sym2d[i][i:]
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out
