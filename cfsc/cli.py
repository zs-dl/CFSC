"""Independent same-report experiments; packaged reference results are never overwritten."""
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import os
import sys
import time



def parser():
    p = argparse.ArgumentParser(description='Run CFSC, marginal SS, and complete-report SS-MLE on identical SS reports.')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--scenario', choices=('uniform', 'zipf', 'spike'), help='Synthetic fixed-input distribution')
    source.add_argument('--input', type=Path, help='One-dimensional .npy or headerless one-column .csv/.txt of integer category codes')
    p.add_argument('--n', type=int, default=None, help='Synthetic sample size (default: 10000); custom input uses all rows')
    p.add_argument('--d', type=int, default=128, help='Complete public domain size, including unobserved categories (default: 128)')
    p.add_argument('--input-seed', type=int, help='Synthetic input seed; defaults to the corresponding paper scenario seed')
    p.add_argument('--zipf-exponent', type=float, default=1.2)
    p.add_argument('--spike-mass', type=float, default=0.6, help='Spike mass on category 0')
    p.add_argument('--epsilon', '--epsilons', dest='epsilons', type=float, nargs='+', default=[2.0])
    perturb = p.add_mutually_exclusive_group()
    perturb.add_argument('--seeds', type=int, nargs='+', help='Mechanism seeds (default: 0)')
    perturb.add_argument('--repetitions', type=int, help='Use mechanism seeds 0 through repetitions-1')
    p.add_argument('--k', type=int, help='Override public subset size; default uses the paper rounding rule per epsilon')
    p.add_argument('--split-seed', type=int, default=20261109)
    p.add_argument('--max-heads', type=int, default=8)
    p.add_argument('--eigen-rtol', type=float, default=1e-10)
    p.add_argument('--mle-gap', type=float, default=1e-11, help='Mean-objective Frank-Wolfe gap tolerance')
    p.add_argument('--mle-max-iterations', type=int, default=5000)
    p.add_argument('--without-mle', action='store_true', help='Run only the four original SS/CFSC outputs')
    p.add_argument('--threads', type=int, default=1, help='BLAS thread limit during estimation (default: 1)')
    p.add_argument('--save-reports', action='store_true', help='Also save generated packed privatized report arrays')
    p.add_argument('--save-states', action='store_true', help='Save CFSC training/application diagnostic arrays')
    p.add_argument('--output', type=Path, required=True, help='New or empty output directory; existing results are never overwritten')
    return p


def sha_bytes(values):
    return hashlib.sha256(values).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, records, fields=None):
    records = list(records)
    fields = fields or (list(records[0]) if records else [])
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def metrics(estimate, target):
    delta = np.asarray(estimate) - target
    return {'mse': float(np.mean(delta * delta)), 'l1': float(np.abs(delta).sum()),
            'linf': float(np.abs(delta).max())}


def aggregate(records, expected, output):
    groups = defaultdict(list)
    for row in records:
        groups[(row['epsilon'], row['estimator'])].append(row)
    summaries = []
    for (eps, estimator), group in sorted(groups.items()):
        row = {k: group[0][k] for k in ['scenario', 'n', 'd', 'k']}
        row.update({'epsilon': eps, 'estimator': estimator, 'n_repetitions': len(group),
                    'expected_repetitions': expected, 'converged_repetitions': sum(x['converged'] for x in group)})
        for metric in ('mse', 'l1', 'linf'):
            x = np.array([r[metric] for r in group])
            row[metric + '_mean'] = float(x.mean())
            row[metric + '_std'] = float(x.std(ddof=1)) if len(x) > 1 else None
        summaries.append(row)
    write_csv(output / 'summary.csv', summaries)
    lookup = {(r['epsilon'], r['seed'], r['estimator']): r for r in records}
    contrasts = []
    for eps in sorted({r['epsilon'] for r in records}):
        for baseline in ('ss_simplex', 'ss_mle'):
            seeds = sorted({r['seed'] for r in records if r['epsilon'] == eps
                            and (eps, r['seed'], baseline) in lookup and (eps, r['seed'], 'cfsc_simplex') in lookup})
            if not seeds:
                continue
            for metric in ('mse', 'l1', 'linf'):
                a = np.array([lookup[eps, seed, baseline][metric] for seed in seeds])
                b = np.array([lookup[eps, seed, 'cfsc_simplex'][metric] for seed in seeds])
                diff = a - b
                lo = hi = None
                if len(seeds) > 1:
                    half = float(student_t.ppf(.975, len(seeds) - 1) * diff.std(ddof=1) / np.sqrt(len(seeds)))
                    lo, hi = float(diff.mean() - half), float(diff.mean() + half)
                contrasts.append({'epsilon': eps, 'candidate': 'cfsc_simplex', 'baseline': baseline,
                                  'metric': metric, 'n_paired': len(seeds), 'baseline_mean': float(a.mean()),
                                  'candidate_mean': float(b.mean()), 'baseline_minus_candidate_mean': float(diff.mean()),
                                  'relative_mean_reduction': float(1 - b.mean() / a.mean()) if a.mean() else None,
                                  'paired_t95_low': lo, 'paired_t95_high': hi})
    write_csv(output / 'paired-contrasts.csv', contrasts)
    return summaries


def run(args):
    if args.input is not None:
        if args.n is not None or args.input_seed is not None:
            raise ValueError('--n and --input-seed apply only to synthetic input')
        values = load_categories(args.input, args.d)
        source = {'kind': 'custom', 'filename': args.input.name}
        scenario = args.input.stem
    else:
        n = 10000 if args.n is None else args.n
        input_seed = SYNTHETIC_SEEDS[args.scenario] if args.input_seed is None else args.input_seed
        values = synthetic_categories(args.scenario, n, args.d, input_seed, args.zipf_exponent, args.spike_mass)
        source = {'kind': 'synthetic', 'scenario': args.scenario, 'seed': input_seed, 'sampler': 'categorical_inverse_cdf_v1',
                  'zipf_exponent': args.zipf_exponent, 'spike_mass': args.spike_mass}
        scenario = args.scenario
    n, d = len(values), args.d
    if args.repetitions is not None and args.repetitions < 1:
        raise ValueError('--repetitions must be positive')
    seeds = list(range(args.repetitions)) if args.repetitions is not None else (args.seeds or [0])
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError('Mechanism seeds must be distinct unsigned 64-bit integers')
    if len(set(args.epsilons)) != len(args.epsilons):
        raise ValueError('Duplicate privacy budgets are not allowed')
    public = {eps: channel_parameters(d, eps, args.k) for eps in args.epsilons}
    config = DecoderConfig(args.split_seed, args.max_heads, args.eigen_rtol)
    if not 0 <= args.split_seed < 2**64 or args.max_heads < 1 or not 0 < args.eigen_rtol < 1:
        raise ValueError('Invalid public split/head/eigen settings')
    if args.mle_gap <= 0 or not np.isfinite(args.mle_gap) or args.mle_max_iterations < 0 or args.threads < 1:
        raise ValueError('Invalid MLE or thread settings')
    mle_config = MLEConfig(gap_tolerance=args.mle_gap, max_iterations=args.mle_max_iterations)
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory; choose a new path to retain existing results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cases').mkdir()
    package = Path(__file__).resolve().parent
    manifest = {'schema_version': 1, 'purpose': 'new_portable_reproduction', 'input': source, 'n': n, 'd': d,
                'input_int64_sha256': sha_bytes(values.astype('<i8', copy=False).tobytes()),
                'epsilons': args.epsilons, 'mechanism_seeds': seeds, 'bit_generator': 'PCG64',
                'decoder': asdict(config), 'mle': None if args.without_mle else asdict(mle_config),
                'blas_thread_limit': args.threads, 'save_reports': args.save_reports, 'save_states': args.save_states,
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                'code_sha256': {p.name: sha_bytes(p.read_bytes()) for p in sorted(package.glob('*.py'))},
                'statistical_unit': 'independent mechanism perturbation conditional on one fixed empirical input',
                'accuracy_uncertainty': 'sample SD (ddof=1); pointwise paired Student-t 95% intervals without multiplicity adjustment'}
    write_json(output / 'manifest.json', manifest)
    rows, failures, unconverged = [], 0, 0
    started = time.monotonic()
    for eps in args.epsilons:
        par = public[eps]
        for seed in seeds:
            case_id = f'eps-{format(eps, ".17g").replace(".", "p")}_seed-{seed:06d}'
            folder = output / 'cases' / case_id
            folder.mkdir()
            meta = {'scenario': scenario, 'n': n, 'd': d, 'k': par['k'], 'epsilon': eps, 'seed': seed}
            try:
                reports = sample_reports(values, d, eps, seed, par['k'])
                # Neither decoder accepts input values or an evaluation target.
                result = decode_reports(reports, d, eps, par['k'], config=config,
                                        mle_config=mle_config, include_mle=not args.without_mle)
                # Evaluation is deliberately separated from report-only estimation.
                target = np.bincount(values, minlength=d).astype(np.float64) / n
                converged = args.without_mle or result['diagnostics']['ss_mle']['converged']
                if not converged:
                    unconverged += 1
                record = {**meta, 'status': 'success' if converged else 'mle_unconverged',
                          'report_sha256': sha_bytes(reports.tobytes()), 'public_channel': par,
                          'metrics': {name: metrics(est, target) for name, est in result['estimates'].items()},
                          'diagnostics': result['diagnostics']}
                np.savez_compressed(folder / 'estimates.npz', empirical_target=target, **result['estimates'])
                if args.save_reports:
                    np.savez_compressed(folder / 'reports.npz', packed_reports=reports)
                if args.save_states:
                    np.savez_compressed(folder / 'states.npz', **result['arrays'])
                for name, score in record['metrics'].items():
                    rows.append({**meta, 'estimator': name, 'converged': bool(converged) if name == 'ss_mle' else True, **score})
            except Exception as exc:
                failures += 1
                record = {**meta, 'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}
            write_json(folder / 'result.json', record)
            write_json(output / 'progress.json', {'completed_cases': sum(1 for p in (output / 'cases').glob('*/result.json')),
                                                 'expected_cases': len(seeds) * len(args.epsilons), 'failed_cases': failures,
                                                 'mle_unconverged_cases': unconverged})
            print(f'{case_id}: {record["status"]}', flush=True)
    write_csv(output / 'repetitions.csv', rows)
    aggregate(rows, len(seeds), output)
    status = {'status': 'completed' if failures == 0 and unconverged == 0 else 'completed_with_failures',
              'expected_cases': len(seeds) * len(args.epsilons), 'failed_cases': failures,
              'mle_unconverged_cases': unconverged, 'elapsed_seconds': time.monotonic() - started}
    write_json(output / 'status.json', status)
    print(json.dumps(status), flush=True)
    return 0 if failures == 0 and unconverged == 0 else 1


def main(argv=None):
    p = parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        p.print_help()
        return 0
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error('--threads must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    # Delayed numerical imports make the process thread limits effective for
    # python -m cfsc and the installed cfsc-run entry point.
    global np, scipy, student_t, decode_reports, SYNTHETIC_SEEDS, channel_parameters
    global load_categories, sample_reports, synthetic_categories, DecoderConfig, MLEConfig
    import numpy as np
    import scipy
    from scipy.stats import t as student_t
    from .api import decode_reports
    from .channel import SYNTHETIC_SEEDS, channel_parameters, load_categories, sample_reports, synthetic_categories
    from .decoder import DecoderConfig
    from .ss_mle import MLEConfig
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
