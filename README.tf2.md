# The (work-in-progress) TF2 extension

This document describes the Tensorflow 2 part-extension of the original `infopt` library.
As of now, it is more of an informal "fork" that only includes a subset of functionalities required for the AFRL 
project.

Please send questions to [YJ](mailto:yjchoe@cmu.edu).

## Installation

All required packages should be installed via `poetry`, as done in [`runs.md#Installation`](runs.md#installation).
The `poetry.lock` and `pyproject.toml` files in this branch (`tf2`) include an additional installation of 
`tensorflow==2.4.1`.

If you've already installed the master version of the package, simply call:
```shell
poetry add tensorflow@^2.4.1
```
Otherwise, follow steps in [`runs.md#Installation`](runs.md#installation), but clone this branch (`tf2`):
```shell
git clone -b tf2 https://github.com/jayanthkoushik/infopt
```

## Getting Started

If you'd like to run additional experiments involving TF2 and/or NN-MCD, 
you'd want to start with [`test_mcdropout_ackley.ipynb`](test_mcdropout_ackley.ipynb) as an example, and 
perhaps modify/add methods in [`exp_sim_optim_simple.py`](exp_sim_optim_simple.py) and 
in [`exputils/objectives.py`](exputils/objectives.py).

If you're only interested in reproducing the original results using NN-INF (in torch), 
checkout the master branch and follow the instructions [here](runs.md) instead.

## List of Code Extensions

All implementations are subject to additional testing. 

- **TF2 port of NN-INF**
    - [`infopt/ihvp_tf.py`](infopt/ihvp_tf.py)
    - [`infopt/nnacq_tf.py`](infopt/nnacq_tf.py)
    - [`infopt/nnmodel_tf.py`](infopt/nnmodel_tf.py)
    - [`exputils/models_tf.py`](exputils/models_tf.py)
    - [`tests/test_ihvp_tf.py`](tests/test_ihvp_tf.py) 
    - (This could, in theory, be incorporated into the experiments code, such as `exp_sim_optim.py`.)
- **Black-Box Optimization with Monte Carlo Dropout (NN-MCD)**
    - [`infopt/nnmodel_mcd_tf.py`](infopt/nnmodel_mcd_tf.py)
- **A "simplified" script for running synthetic experiments interactively (and debugging)**
    - [`exp_sim_optim_simple.py`](exp_sim_optim_simple.py): a smaller version of [`exp_sim_optim.py`](exp_sim_optim.py)
    - [`exputils/objectives.py`](exputils/objectives.py): for potentially adding new objectives, e.g., RandomNN
- **Some test runs / playgrounds**
    - GP-LCB vs. NN-INF vs. NN-MCD
        - [`test_mcdropout_ackley.ipynb`](test_mcdropout_ackley.ipynb)
        - [`test_mcdropout_randomnn.ipynb`](test_mcdropout_randomnn.ipynb)
    - NN-INF (torch) vs. NN-INF (tf2)
        - [`test_torch2tf_ackley.ipynb`](test_torch2tf_ackley.ipynb)
        - [`test_torch2tf_randomnn.ipynb`](test_torch2tf_randomnn.ipynb)
    

