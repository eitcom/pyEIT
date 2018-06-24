# EIT Reconstruction Algorithms

This folder contains EIT forward and inverse solvers.

## Forward Solver

`fem.py` is the forward solver using the Finite Element Method. A unstructured triangular (2D) or tetrahedron (3D) mesh data structure is the input to this module. Two main functions, named `solve_once`  and `solve` are provided. `solve_once` calculates the distribution of voltages `u` given 1 stimulation pattern. `solve` solves a complete EIT problem where the stimulation patterns (2 electrodes only) are provided by `ex_mat`.

## Inverse Solver (a.k.a EIT Imaging Algorithms)

`base.py` is the root class for an EIT algorithm. It builds up the Jacobian matrix as well as other regularization parameters and matrix. Subsequent EIT algorithms build upon this module.

- `bp.py` contains Back projection and filtered BP.
- `jac.py` The traditional Gauss Newton (GN) methods. It uses `lm`, `kotre` and other customized regularization matrix.
- `greit.py` is the GREIT algorithm with statistical spatial filtering and EIT imaging matrix. Note that training a imaging matrix from real-life measurements (training datasets) is not yet implemented. GREIT can be used as a spatial interpolation (from unstructured meshes to uniform grids) of GN methods, if `W_s` and `W_n` are properly integrated.

You may help to improve these modules by starting a issue or a pull request.

## Pre and Post processing

There are some modules for post or pre processing purpose.

- `utils.py` provides `eit_scan_lines` module. It is a simple function provided as is. You can build other stacked stimulation patterns upon this.
- `interp2d.py` comprehensive interpolation modules. This function serves as two purpose, 1, interpolate from unstructured mesh structures (triangles) into uniform or structured grids. 2, interpolate within unstructured meshes, for example, maps values on elements to values on nodes and vice versa.

## TODOs

More algorithms and complete electrode models (CEM) modules need to be implemented.