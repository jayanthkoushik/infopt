# Reproducing `infopt` Results

## Installation

Supported Python versions: 3.7.x or 3.8.x.

1. Install [`poetry`](https://python-poetry.org/docs/):
    ```shell
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```
2. Clone this repository and install dependencies via `poetry`:
    ```shell
    git clone https://github.com/jayanthkoushik/infopt
    cd infopt
    poetry install --extras "test experiments"
    ```
3. Use `poetry`'s virtual environment:
    ```shell
    poetry shell
    ```

## Sample Runs

Run experiments:
```shell
# NN-INF on Ackley 2D
python exp_sim_optim.py run \
    --fname ackley --fdim 2 --init-points 1 --optim-iters 100 \
    --save-file results/ackley2d/nn001.pkl --tb-dir logdir/ackley2d/nn001 \
    nn --layer-sizes 8,8,4
# GP-LCB on Ackley 2D
python exp_sim_optim.py run \
    --fname ackley --fdim 2 --init-points 1 --optim-iters 100 \
    --save-file results/ackley2d/gp001.pkl --tb-dir logdir/ackley2d/gp001 \
    gp --acq-type lcb
# ...
```

Plot regrets (together):
```shell
python exp_sim_optim.py plot-regrets \
    --res-dir results/ackley2d --save-file results/ackley2d/regrets.pdf
```

## Hyperparameter Options

```text
$ python exp_sim_optim.py run --help
arguments:
  mname None ...                           ({gp / nn / nr})

options:
  -h, --help                               show this help message and exit (optional)
  --save-file file                         (required)
  --fname function                         ({ackley / alpine1 / alpine2 / beale / branin / dropwave / eggholder /
                                             forrester / powers / rastrigin / rosenbrock / sixhumpcamel} required)
  --fdim int                               (required)
  --tb-dir dir                             (optional)

optimization parameters:
  --init-points int                        (default: 1)
  --optim-iters int                        (default: 1)
  --model-update-interval int              (default: 1)

exploration weight configuration:
  --exp-multiplier float                   (default: 0.1)
  --exp-gamma float                        (default: 0.1)
  --use-const-exp-w float                  (optional)

evaluator for selecting points:
  --evaluator-cls evaluator                (default: Sequential)
  --evaluator-params key=value,[...]       (default: {})

output noise:
  --gaussian-noise-with-scale float        (optional)
  --gprbf-noise-with-scale float           (optional)
```

```text
$ python exp_sim_optim.py run nn --help                               
options:
  -h, --help                           show this help message and exit (optional)
  --layer-sizes int,int[,...]          (required)

model low-rank ihvp:
  --ihvp-rank int                      (default: 10)
  --ihvp-batch-size int                (default: 8)
  --ihvp-optim-cls optimizer           (default: Adam)
  --ihvp-optim-params key=value,[...]  (default: {'lr':␣0.01})
  --ihvp-ckpt-every int                (default: 25)
  --ihvp-iters-per-point int           (default: 25)
  --ihvp-loss-cls loss                 (default: MSELoss)

model wrapper for gpyopt:
  --bom-optim-cls optimizer            (default: Adam)
  --bom-optim-params key=value,[...]   (default: {'lr':␣0.02})
  --bom-up-batch-size int              (default: 16)
  --bom-up-iters-per-point int         (default: 50)
  --bom-n-higs int                     (default: 8)
  --bom-ihvp-n int                     (default: 15)
  --bom-ckpt-every int                 (default: 25)
  --bom-loss-cls cls                   (default: MSELoss)

lcb acquisition:
  --acq-optim-cls optimizer            (default: Adam)
  --acq-optim-params key=value,[...]   (default: {'lr':␣0.05})
  --acq-optim-iters int                (default: 5000)
  --acq-optim-lr-decay-step-size int   (default: 1000)
  --acq-optim-lr-decay-gamma float     (default: 0.2)
  --acq-ckpt-every int                 (default: 1000)
  --acq-rel-tol float                  (default: 0.001)
  --no-acq-reinit-optim-start         (default: True)
```

