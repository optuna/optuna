import bisect
import numpy as np

from optuna.study._multi_objective import _is_pareto_front


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 2
    rect_diag_y = np.concatenate([reference_point[1:], sorted_pareto_sols[:-1, 1]])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y


def _compute_3d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume in 3D using a sweep-line algorithm.
    Time complexity is O(N^2) in worst case due to list insertions, but O(N) memory.
    This is much more memory efficient than the vectorized O(N^2) approach and faster for large N.
    """
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 3
    n = sorted_pareto_sols.shape[0]

    # Sort by z (dim 2) ascending
    z_order = np.argsort(sorted_pareto_sols[:, 2])
    sorted_by_z = sorted_pareto_sols[z_order]

    staircase = []  # List of [x, y]
    area = 0.0
    total_volume = 0.0
    prev_z = sorted_by_z[0][2]

    Rx = reference_point[0]
    Ry = reference_point[1]
    Rz = reference_point[2]

    for i in range(n):
        p = sorted_by_z[i]
        z = p[2]

        total_volume += area * (z - prev_z)
        prev_z = z

        x, y = p[0], p[1]

        # Find insertion point for x
        # staircase is sorted by x. We use [x, inf] to find the first element with e.x > x.
        idx = bisect.bisect_right(staircase, [x, float("inf")])

        # Check dominance by existing (to the left)
        if idx > 0:
            if staircase[idx - 1][1] <= y:
                continue

        # Find range of dominated points (to the right)
        end_idx = idx
        while end_idx < len(staircase) and staircase[end_idx][1] >= y:
            end_idx += 1

        # Remove contributions of dominated points
        for k in range(idx, end_idx):
            e = staircase[k]
            nxt = staircase[k + 1] if k + 1 < len(staircase) else None
            nx = nxt[0] if nxt else Rx
            area -= (nx - e[0]) * (Ry - e[1])

        # Update contribution of the point to the left
        if idx > 0:
            prev = staircase[idx - 1]
            old_next_x = staircase[idx][0] if idx < len(staircase) else Rx
            area -= (old_next_x - prev[0]) * (Ry - prev[1])
            area += (x - prev[0]) * (Ry - prev[1])

        # Add contribution of new point
        new_next_x = staircase[end_idx][0] if end_idx < len(staircase) else Rx
        area += (new_next_x - x) * (Ry - y)

        # Update staircase
        staircase[idx:end_idx] = [[x, y]]

    total_volume += area * (Rz - prev_z)

    return total_volume


def _compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    if sorted_loss_vals.shape[0] == 1:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        inclusive_hv = 1.0
        for r, v in zip(reference_point, sorted_loss_vals[0]):
            inclusive_hv *= r - v
        return float(inclusive_hv)
    elif sorted_loss_vals.shape[0] == 2:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        hv1, hv2, intersec = 1.0, 1.0, 1.0
        for r, v1, v2 in zip(reference_point, sorted_loss_vals[0], sorted_loss_vals[1]):
            hv1 *= r - v1
            hv2 *= r - v2
            intersec *= r - max(v1, v2)
        return hv1 + hv2 - intersec

    inclusive_hvs = (reference_point - sorted_loss_vals).prod(axis=-1)
    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    limited_sols_array = np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)
    return inclusive_hvs[-1] + sum(
        _compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hvs[i], reference_point)
        for i in range(inclusive_hvs.size - 1)
    )


def _compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    assert limited_sols.shape[0] >= 1
    if limited_sols.shape[0] <= 3:
        # NOTE(nabenabe): Don't use _is_pareto_front for 3 or fewer points to avoid its overhead.
        return inclusive_hv - _compute_hv(limited_sols, reference_point)

    # NOTE(nabenabe): The following holds only for an incremental version of `_is_pareto_front_nd`,
    # meaning that if there are duplicated Pareto solutions, the second one must be judged as
    # weakly dominated by the first one.
    # NOTE(nabenabe): As the following line is a hack for speedup, I will describe several
    # important points to note. Even if we do not run _is_pareto_front below or use
    # assume_unique_lexsorted=False instead, the result of this function does not change, but this
    # function simply becomes slower.
    #
    # For simplicity, I call an array ``quasi-lexsorted`` if it is sorted by the first objective.
    #
    # Reason why it will be faster with _is_pareto_front
    #   Hypervolume of a given solution set and a reference point does not change even when we
    #   remove non Pareto solutions from the solution set. However, the calculation becomes slower
    #   if the solution set contains many non Pareto solutions. By removing some obvious non Pareto
    #   solutions, the calculation becomes faster.
    #
    # Reason why assume_unique_lexsorted must be True for _is_pareto_front
    #   assume_unique_lexsorted=True actually checks weak dominance and solutions will be weakly
    #   dominated if there are duplications, so we can remove duplicated solutions by this option.
    #   In other words, assume_unique_lexsorted=False may significantly slow down when limited_sols
    #   has many duplicated Pareto solutions because this function becomes an exponential algorithm
    #   without duplication removal.
    #
    # NOTE(nabenabe): limited_sols can be non-unique and/or non-lexsorted, so I will describe why
    # it is fine.
    #
    # Reason why we can specify assume_unique_lexsorted=True even when limited_sols is not
    #   All ``False`` in on_front will be correct (, but it may not be the case for ``True``) even
    #   if limited_sols is not unique or not lexsorted as long as limited_sols is quasi-lexsorted,
    #   which is guaranteed. As mentioned earlier, if all ``False`` in on_front is correct, the
    #   result of this function does not change.
    on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=True)
    return inclusive_hv - _compute_hv(limited_sols[on_front], reference_point)


def compute_hypervolume(
    loss_vals: np.ndarray, reference_point: np.ndarray, assume_pareto: bool = False
) -> float:
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension.
    For 3 dimensions or higher, the WFG algorithm will be used.
    Please refer to ``A Fast Way of Calculating Exact Hypervolumes`` for the WFG algorithm.

    .. note::
        This class is used for computing the hypervolumes of points in multi-objective space.
        Each coordinate of each point represents a ``values`` of the multi-objective function.

    .. note::
        We check that each objective is to be minimized. Transform objective values that are
        to be maximized before calling this class's ``compute`` method.

    Args:
        loss_vals:
            An array of loss value vectors to calculate the hypervolume.
        reference_point:
            The reference point used to calculate the hypervolume.
        assume_pareto:
            Whether to assume the Pareto optimality to ``loss_vals``.
            In other words, if ``True``, none of loss vectors are dominated by another.
            ``assume_pareto`` is used only for speedup and it does not change the result even if
            this argument is wrongly given. If there are many non-Pareto solutions in
            ``loss_vals``, ``assume_pareto=True`` will speed up the calculation.

    Returns:
        The hypervolume of the given arguments.

    """

    if not np.all(loss_vals <= reference_point):
        raise ValueError(
            "All points must dominate or equal the reference point. "
            "That is, for all points in the loss_vals and the coordinate `i`, "
            "`loss_vals[i] <= reference_point[i]`."
        )
    if not np.all(np.isfinite(reference_point)):
        # reference_point does not have nan, thanks to the verification above.
        return float("inf")
    if loss_vals.size == 0:
        return 0.0

    if not assume_pareto:
        unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
        sorted_pareto_sols = unique_lexsorted_loss_vals[on_front]
    else:
        # NOTE(nabenabe): The result of this function does not change both by
        # np.argsort(loss_vals[:, 0]) and np.unique(loss_vals, axis=0).
        # But many duplications in loss_vals significantly slows down the function.
        # TODO(nabenabe): Make an option to use np.unique.
        sorted_pareto_sols = loss_vals[loss_vals[:, 0].argsort()]

    if reference_point.shape[0] == 2:
        hv = _compute_2d(sorted_pareto_sols, reference_point)
    elif reference_point.shape[0] == 3:
        # NOTE: For 3D points, we always prefer _compute_3d to _compute_hv because the time
        # complexity of _compute_3d is O(N^2), while that of _compute_nd is \\Omega(N^3)
        # - It calls _compute_exclusive_hv with i points for i = 0, 1, ..., N-1
        # - _compute_exclusive_hv calls _is_pareto_front, which is quadratic
        #   with the number of points
        hv = _compute_3d(sorted_pareto_sols, reference_point)
    else:
        hv = _compute_hv(sorted_pareto_sols, reference_point)

    # NOTE(nabenabe): `nan` happens when inf - inf happens, but this is inf in hypervolume due to
    # the submodularity.
    return hv if np.isfinite(hv) else float("inf")
