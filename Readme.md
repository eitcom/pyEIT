# ![pyeit](figs/logo.png)

A python-based, open-source package for Electrical Impedance Tomography (EIT)

**Dependencies:**

 - numpy (tested with `numpy-1.10.4`)
 - scipy (tested with `scipy-0.17.0`)
 - matplotlib (tested with `matplotlib-1.5.1`)
 - pandas (tested with `pandas-0.17.1`)
 - xarray (optional, for large scale data analysis)
 - distmesh or meshpy (both are optional), it currently has a build-in distmesh2d module
 - tetgen (optional) for generating 3D meshes

The distribution, [Anaconda from continuum](https://www.continuum.io/downloads), is suggested to be used with this package.

**Currently suppots:**

 - 2D forward and inverse computing of EIT
 - Reconstruction algorithms : Gauss-Newton solver (JAC), Back-projection (BP), 2D GREIT
 - 2D visualization code

## Demos

## Todos

 - [ ] Add support for 3D forward and inverse computing
 - [ ] Port iso2mesh to python, and use MRI data as input for 3D mesh generation
 - [ ] More algorithms and data pre-processing modules

## Version

 - alpha, 2016-03-07
