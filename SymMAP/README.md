# SymMAP (Symbolic Matrix Preconditioning)

SymMAP is a novel Recurrent Neural Network (RNN)-based symbolic discovery framework that searches for symbolic expressions of efficient preconditioning parameters.

## Features

- **Automated Discovery**: Automatically generates symbolic expressions for preconditioning parameters that can be applied to a wide range of matrices.
- **RNN Architecture**: Leverages a sophisticated RNN model to learn from historical data and improve preconditioning strategies over time.
- **Scalability**: Designed to scale with increasing matrix sizes and complexity, ensuring robust performance across different computational environments.

## Code for SymMAP

The SymMAP framework is constructed based on [Deep Symbolic Optimization](https://github.com/dso-org/deep-symbolic-optimization/tree/master). The main components of the SymMAP framework are as follows:

- `dso/run.py`: Main script to run the SymMAP framework.
- `json2multicsv.py`: Converts JSON files to multiple CSV files for training and testing.
- `dso/core.py`: Core deep symbolic optimizer construct.
- `dso/train.py`: Defines main training loop for deep symbolic optimization.
- `dso/const.py`: Constant optimizer used for deep symbolic optimization.

The main directories are:
- `config`: Configuration files for the SymMAP framework.
- `dso/policy_optimizer`: Provides several policy optimizers for the deep symbolic optimization framework, including PG, PPO, and PQT.
- `dso/task/regression`: Provides regression tasks for the deep symbolic optimization framework.

## Running SymMAP
To run SymMAP, run

```bash
python -m dso.run config/myconfig.json
```

Here, `config/myconfig.json` is one of the configuration files.