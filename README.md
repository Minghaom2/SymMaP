# SymMaP: Symbolic Matrix Preconditioning (Neurips 2025)

SymMaP is a symbolic matrix preconditioning framework that learns compact, human-interpretable formulas for selecting preconditioning parameters in linear solvers. It constructs supervision via adaptive grid search, and trains an RNN with a risk-seeking objective to generate prefix-form expressions that map problem features to near-optimal parameters. The learned one-line expressions plug directly into CPU-oriented solver stacks (e.g., PETSc) with negligible overhead, improving time-to-solution and conditioning across SOR/SSOR/AMG settings while preserving transparency and ease of deployment.

### Getting Started

To begin, you need to install the ``dso`` package in an environment with Python 3.6 or higher. Navigate to ``SymMaP/`` and run:

```python
pip install -e ./dso
```


you need to generate a dataset that includes the equations and their corresponding optimal preconditioning parameters. As an example, we consider a second-order elliptic PDE preconditioned using the SOR method.

Navigate to the data generation directory:

``cd data/sor``

Run the data generation script:

``python generate.py``

To use a different equation or preconditioning method, or to adjust the generation parameters, modify the relevant sections in generate.py or navigate to other preprocessor name directories.

The generated dataset will be saved in the form of ``X.json`` and ``y.json``.

### Training



### Results



### Code Structure

<pre>
├─ SymMaP/
│  ├─ 
│  ├─ 
│  └─ 
├─ data/
│  └─ precondition/
│     ├─ e.c              # PETSc equation solver file
│     ├─ makefile         # CMake configuration file
│     └─ generate.py      # Python file for dataset generation
└─ README.md   
</pre>

### Citation
