# SymMaP: Improving Computational Efficiency in Linear Solvers through Symbolic Preconditioning

### Getting Started

To begin, you need to generate a dataset that includes the equations and their corresponding optimal preconditioning parameters. As an example, we consider a second-order elliptic PDE preconditioned using the SOR method.

Navigate to the data generation directory:

``cd data/sor``

Run the data generation script:

``python generate.py``

To use a different equation or preconditioning method, or to adjust the generation parameters, modify the relevant sections in generate.py or navigate to other preprocessor name directories.

The generated dataset will be saved in the form of X.json and y.json.

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
