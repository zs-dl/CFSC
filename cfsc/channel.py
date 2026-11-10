"""Portable SS channel and fixed-input generation for research reproduction.

The sampling sequence follows the original framework's fixed-size SS sampler:
all inclusion decisions first, followed by chunked random priorities. This
module is an export adapter; decoder.py and ss_mle.py are frozen implementations.
"""
import math
from pathlib import Path
import numpy as np

SYNTHETIC_SEEDS = {'uniform': 2026110501, 'zipf': 2026110503, 'spike': 2026110504}


def validate_categories(values, d):
    x = np.asarray(values)
    if type(d) is not int or d < 2:
        raise ValueError('The complete public domain size d must be an integer >= 2')
    if x.ndim != 1 or x.size < 2:
        raise ValueError('Input must be a one-dimensional sequence with at least two users')
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Categories must be integer codes; no implicit rounding is performed')
    if np.any(x < 0) or np.any(x >= d):
        raise ValueError('Every category must be in [0, d); retain unobserved public categories in d')
    return np.ascontiguousarray(x, dtype=np.int64)


def channel_parameters(d, epsilon, k=None):
    if type(d) is not int or d < 2 or not math.isfinite(epsilon) or not 0 < epsilon <= 50:
        raise ValueError('Require d >= 2 and finite 0 < epsilon <= 50')
    z = math.exp(-epsilon)
    if k is None:
        k = max(1, round(d * z / (1 + z)))
    if type(k) is not int or not 1 <= k < d:
        raise ValueError('Subset size must satisfy 1 <= k < d')
    p = k / (k + (d - k) * z)
    q = (k - p) / (d - 1)
    return {'d': d, 'epsilon': float(epsilon), 'k': k, 'p': p, 'q': q}


def sample_reports(values, d, epsilon, seed=0, k=None):
    """Return packed little-endian SS bitmaps, with zero padding beyond d."""
    x = validate_categories(values, d)
    public = channel_parameters(d, epsilon, k)
    k, p = public['k'], public['p']
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Mechanism seed must be an unsigned 64-bit integer')
    rng = np.random.Generator(np.random.PCG64(seed))
    reports = np.empty((x.size, (d + 7) // 8), dtype=np.uint8)
    included_all = rng.random(x.size) < p
    chunk_rows = max(1, (1 << 19) // d)
    for start in range(0, x.size, chunk_rows):
        selected = x[start:start + chunk_rows]
        rows = np.arange(selected.size)
        included = included_all[start:start + selected.size]
        if k == 1:
            other = rng.integers(d - 1, size=selected.size)
            chosen = np.where(included, selected, other + (other >= selected))[:, None]
        else:
            priorities = rng.random((selected.size, d))
            priorities[rows, selected] = np.inf
            chosen = np.argpartition(priorities, k - 1, axis=1)[:, :k]
            chosen[included, -1] = selected[included]
        bits = np.zeros((selected.size, d), dtype=bool)
        bits[rows[:, None], chosen] = True
        reports[start:start + selected.size] = np.packbits(bits, axis=1, bitorder='little')
    return reports


def synthetic_categories(scenario, n=10000, d=128, seed=None, zipf_exponent=1.2, spike_mass=0.6):
    if scenario not in SYNTHETIC_SEEDS:
        raise ValueError('Scenario must be uniform, zipf, or spike')
    if type(n) is not int or n < 2 or type(d) is not int or d < 2:
        raise ValueError('Require integer n >= 2 and d >= 2')
    seed = SYNTHETIC_SEEDS[scenario] if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Input seed must be an unsigned 64-bit integer')
    if scenario == 'uniform':
        weights = np.ones(d, dtype=np.float64)
    elif scenario == 'zipf':
        if not math.isfinite(zipf_exponent) or zipf_exponent <= 0:
            raise ValueError('Zipf exponent must be positive and finite')
        weights = np.power(np.arange(1, d + 1, dtype=np.float64), -zipf_exponent)
    else:
        if not math.isfinite(spike_mass) or not 0 < spike_mass <= 1:
            raise ValueError('Spike mass must be in (0, 1]')
        weights = np.full(d, (1 - spike_mass) / (d - 1), dtype=np.float64)
        weights[0] = spike_mass
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    uniforms = np.random.Generator(np.random.PCG64(seed)).random(n)
    return np.searchsorted(cdf, uniforms, side='right').astype(np.int64, copy=False)


def load_categories(path, d):
    """Load a .npy vector or a headerless, single-column integer CSV/text file."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in ('.csv', '.txt'):
        # Parse strings first so a malformed value such as 2.5 is never truncated.
        tokens = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None,
                            dtype=str, ndmin=2)
        if tokens.shape[1] != 1:
            raise ValueError('CSV/text input must contain exactly one category per row, without a header')
        try:
            values = np.array([int(t) for t in tokens[:, 0]], dtype=np.int64)
        except (ValueError, OverflowError) as exc:
            raise ValueError('CSV/text categories must be integer codes, without a header') from exc
    else:
        raise ValueError('Input extension must be .npy, .csv, or .txt')
    return validate_categories(values, d)
