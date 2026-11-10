# CFSC

Run CFSC, SS and SS-MLE on the same locally private subset reports. Requires Python 3.10+.

## Run

```sh
python -m pip install -r requirements.txt
python run.py
```

The default runs a small Zipf example and prints its output directory. Open `summary.csv` there for MSE, L1 and Linf.

Use an included dataset:

```sh
python run.py --input data/adult.npy --d 16 --epsilon 2 --seeds 0 1 2 --output outputs/adult
```

Run several budgets and repetitions:

```sh
python run.py --input data/zipf.npy --d 128 --epsilons 0.5 1 2 3 4 5 --repetitions 100 --output outputs/zipf
```

Use a new output directory each time. `python run.py --help` lists all options. Custom input can be a one-dimensional integer `.npy` array or a headerless single-column CSV; codes must be in `[0,d)`.

## Included files

- `cfsc/`, `run.py`: runnable code.
- `data/*.npy`: six category-coded datasets. `datasets.json` gives category order and source attribution.
- `results/accuracy.csv`: recorded MSE/L1/Linf means and sample standard deviations across six datasets, six budgets and seven methods. The runner itself implements SS/CFSC/SS-MLE.
- `results/runtime.csv`: recorded decoder-time medians and ranges, in milliseconds.

| Dataset | Records | `--d` |
| --- | ---: | ---: |
| adult | 32561 | 16 |
| bank | 45211 | 12 |
| uniform | 10000 | 128 |
| zipf | 10000 | 128 |
| census | 20000 | 400 |
| spike | 10000 | 128 |

Metrics target the fixed input's empirical frequencies: MSE averages squared coordinate errors, L1 sums absolute errors, and Linf takes the maximum. The supplied accuracy summaries use 100 mechanism seeds; timing summaries use five measurements per condition.

Adult, Bank Marketing and US Census Data (1990) are from UCI under CC BY 4.0. The included vectors retain only the selected categorical fields; attribution and transformations are recorded in `data/datasets.json`.
