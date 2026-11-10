"""Maximum likelihood from complete, fixed-cardinality SS reports.

The optimization uses public channel parameters and privatized reports only.
It maximizes mean(log(1 + expm1(epsilon) * I @ v)) on the simplex.
The omitted channel normalizer is constant in v.  The Frank--Wolfe gap
max(gradient) - dot(v, gradient) bounds the objective suboptimality globally.
"""
from dataclasses import asdict, dataclass
from math import comb
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class MLEConfig:
    gap_tolerance: float = 1e-9
    max_iterations: int = 5000
    group_max_combinations: int = 200000
    armijo: float = 1e-4
    max_backtracks: int = 60


def project_simplex(values):
    """Euclidean projection onto the unit simplex without dropping categories."""
    x = np.asarray(values, dtype=np.float64)
    ordered = np.sort(x)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    active = np.flatnonzero(ordered > cumulative / np.arange(1, x.size + 1))
    theta = cumulative[active[-1]] / (active[-1] + 1)
    result = np.maximum(x - theta, 0.0)
    result /= result.sum()
    return result


def _validate_packed(reports, n, d, k):
    reports = np.asarray(reports)
    if reports.dtype != np.uint8 or reports.shape != (n, (d + 7) // 8):
        raise ValueError("Expected packed uint8 reports of shape (n, ceil(d/8))")
    if n < 1 or d < 2 or not 1 <= k < d:
        raise ValueError("Require n >= 1, d >= 2 and 1 <= k < d")
    popcount = np.array([int(x).bit_count() for x in range(256)], dtype=np.uint8)
    if not np.all(popcount[reports].sum(axis=1, dtype=np.int64) == k):
        raise ValueError("Report cardinality does not match public k")
    if d % 8 and np.any(reports[:, -1] >> (d % 8)):
        raise ValueError("Nonzero padding bits outside the public domain")
    return reports


def _separable_solution(probabilities, c, complement=False):
    """Exact water-filling solution for singleton or co-singleton reports."""
    p = np.asarray(probabilities, dtype=np.float64)
    if not complement:
        sorted_p = np.sort(p)[::-1]
        cumulative = np.cumsum(sorted_p)
        lam = cumulative / (1.0 + np.arange(1, p.size + 1) / c)
        active = np.flatnonzero(sorted_p > lam / c)
        dual = lam[active[-1]]
        v = np.maximum(p / dual - 1.0 / c, 0.0)
    else:
        zeros = p == 0.0
        if np.any(zeros):
            v = zeros.astype(float) / zeros.sum()
        else:
            alpha = c / (1.0 + c)
            sorted_p = np.sort(p)
            cumulative = np.cumsum(sorted_p)
            lam = cumulative / (np.arange(1, p.size + 1) / alpha - 1.0)
            active = np.flatnonzero(sorted_p < lam / alpha)
            dual = lam[active[-1]]
            v = np.maximum(1.0 / alpha - p / dual, 0.0)
    return v / v.sum()


class ReportObjective:
    """Grouped observed-subset objective; rows retain their exact frequencies."""
    def __init__(self, matrix, frequencies, epsilon):
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.c = float(np.expm1(epsilon))
        self.function_evaluations = 0
        self.gradient_evaluations = 0

    def value_gradient(self, v):
        denominators = 1.0 + self.c * (self.matrix @ v)
        if np.any(denominators <= 0.0):
            raise ArithmeticError("Likelihood denominator is not positive")
        value = float(self.frequencies @ np.log(denominators))
        gradient = self.c * (self.matrix.T @ (self.frequencies / denominators))
        self.function_evaluations += 1
        self.gradient_evaluations += 1
        return value, gradient

    def value(self, v):
        denominators = 1.0 + self.c * (self.matrix @ v)
        self.function_evaluations += 1
        if np.any(denominators <= 0.0):
            return -np.inf
        return float(self.frequencies @ np.log(denominators))


def _spectral_projected_gradient(objective, initial, k, config):
    """Monotone spectral projected gradient with safeguarded BB steps.

    Every accepted point is feasible. Armijo uses the actual concave objective,
    and termination is based on the global first-order gap, not iterate change.
    """
    v = initial.copy()
    value, gradient = objective.value_gradient(v)
    d = v.size
    c = objective.c
    t_uniform = 1.0 + c * k / d
    curvature = c * c * k * (1.0 - k / d) / ((d - 1) * t_uniform * t_uniform)
    step_scale = 1.0 / max(curvature, 1e-16)
    iterations = 0
    total_backtracks = 0
    status = "iteration_limit"
    initial_value = value
    gap = float(np.max(gradient) - v @ gradient)
    for iteration in range(config.max_iterations + 1):
        gap = max(0.0, float(np.max(gradient) - v @ gradient))
        if gap <= config.gap_tolerance:
            status = "optimality_gap"
            break
        if iteration == config.max_iterations:
            break
        # Subtracting a common gradient component changes no simplex projection.
        centered_gradient = gradient - np.max(gradient)
        direction = project_simplex(v + step_scale * centered_gradient) - v
        directional_derivative = float(centered_gradient @ direction)
        if directional_derivative <= 0.0:
            status = "numerical_stationarity_without_certificate"
            break
        fraction = 1.0
        accepted = False
        # Floating-point slack only affects line-search acceptance, never the gap.
        slack = 8.0 * np.finfo(float).eps * max(1.0, abs(value))
        for backtrack in range(config.max_backtracks):
            trial = v + fraction * direction
            trial = np.maximum(trial, 0.0)
            trial /= trial.sum()
            trial_value = objective.value(trial)
            if trial_value + slack >= value + config.armijo * fraction * directional_derivative:
                accepted = True
                break
            fraction *= 0.5
        total_backtracks += backtrack
        if not accepted:
            status = "line_search_failure"
            break
        trial_value, trial_gradient = objective.value_gradient(trial)
        displacement = trial - v
        gradient_difference = trial_gradient - gradient
        denominator = -float(displacement @ gradient_difference)
        if denominator > 0.0:
            step_scale = float(np.clip((displacement @ displacement) / denominator, 1e-12, 1e12))
        else:
            step_scale = min(step_scale * 2.0, 1e12)
        v, value, gradient = trial, trial_value, trial_gradient
        iterations = iteration + 1
    return v, {
        "converged": bool(gap <= config.gap_tolerance),
        "termination": status,
        "fw_gap": gap,
        "objective": value,
        "initial_objective": initial_value,
        "iterations": iterations,
        "backtracks": total_backtracks,
        "function_evaluations": objective.function_evaluations,
        "gradient_evaluations": objective.gradient_evaluations,
    }


def solve_ss_mle(reports, n, d, epsilon, k, config=None):
    """Decode packed complete SS reports using a certified simplex MLE.

    Returns ``estimate`` as a float64 vector and JSON-serializable diagnostics.
    No original categories, evaluation target or dataset identity are accepted.
    """
    if config is None:
        config = MLEConfig()
    elif isinstance(config, dict):
        config = MLEConfig(**config)
    if not isinstance(config, MLEConfig):
        raise TypeError("config must be MLEConfig, a matching dict, or None")
    if config.gap_tolerance <= 0 or config.max_iterations < 0 or config.max_backtracks < 1:
        raise ValueError("Invalid optimization tolerances or iteration limits")
    if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > 50:
        raise ValueError("Require finite 0 < epsilon <= 50")
    started = perf_counter()
    reports = _validate_packed(reports, n, d, k)
    c = float(np.expm1(epsilon))
    if k in (1, d - 1):
        bits = np.unpackbits(reports, axis=1, bitorder="little")[:, :d]
        counts = bits.sum(axis=0, dtype=np.int64)
        if k == d - 1 and k != 1:
            counts = n - counts
        frequencies = counts.astype(float) / n
        preprocessing_seconds = perf_counter() - started
        optimization_start = perf_counter()
        complement = k != 1
        v = _separable_solution(frequencies, c, complement=complement)
        if complement:
            denominator = 1.0 + c * (1.0 - v)
            gradient = -c * frequencies / denominator
            solver = "exact_complement_water_filling"
        else:
            denominator = 1.0 + c * v
            gradient = c * frequencies / denominator
            solver = "exact_singleton_water_filling"
        value = float(frequencies @ np.log(denominator))
        gap = max(0.0, float(np.max(gradient) - v @ gradient))
        diagnostics = {
            "converged": bool(gap <= config.gap_tolerance),
            "termination": "closed_form",
            "fw_gap": gap,
            "objective": value,
            "iterations": 0,
            "function_evaluations": 1,
            "gradient_evaluations": 1,
            "unique_reports": int(np.count_nonzero(counts)),
            "grouped": True,
            "solver": solver,
        }
    else:
        grouped = comb(d, k) <= config.group_max_combinations
        if grouped:
            unique, multiplicities = np.unique(reports, axis=0, return_counts=True)
            matrix = np.unpackbits(unique, axis=1, bitorder="little")[:, :d]
            frequencies = multiplicities.astype(float) / n
        else:
            matrix = np.unpackbits(reports, axis=1, bitorder="little")[:, :d]
            frequencies = np.full(n, 1.0 / n)
        objective = ReportObjective(matrix, frequencies, epsilon)
        marginal = objective.frequencies @ objective.matrix
        inclusion_true = k * (1.0 + c) / (d + k * c)
        inclusion_other = (k - inclusion_true) / (d - 1)
        initial = project_simplex((marginal - inclusion_other) / (inclusion_true - inclusion_other))
        preprocessing_seconds = perf_counter() - started
        optimization_start = perf_counter()
        v, diagnostics = _spectral_projected_gradient(objective, initial, k, config)
        diagnostics.update({
            "unique_reports": int(objective.matrix.shape[0]) if grouped else None,
            "objective_rows": int(objective.matrix.shape[0]),
            "grouped": bool(grouped),
            "solver": "spectral_projected_gradient",
            "initialization": "SS_marginal_simplex",
        })
    diagnostics.update({
        "n": int(n), "d": int(d), "k": int(k), "epsilon": float(epsilon),
        "preprocessing_seconds": float(preprocessing_seconds),
        "optimization_seconds": float(perf_counter() - optimization_start),
        "total_seconds": float(perf_counter() - started),
        "simplex_sum_error": float(abs(v.sum() - 1.0)),
        "minimum_coordinate": float(v.min()),
        "config": asdict(config),
        "objective_normalization": "mean log unnormalized report probability; constant -log(N) omitted",
    })
    return {"estimate": v, "diagnostics": diagnostics}
