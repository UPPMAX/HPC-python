---
tags:
  - Python
  - bundles
  - software
  - module
  - torch
  - PyTorch
  - Matplotlib
---


# Python bundles


## Introduction

- The bundle names reflect the content, like Python packages, and its version, but also which Python version, compilers and libraries that are compatible with it.

- The module endings may contain GCCcore-X.Y.Z and/or [YEAR-a/b]. Example ``SciPy-bundle/2024.05-gfbf-2024a`` or ``Python/3.12.3-GCCcore-13.3.0``
    - GCCcore reflects the GCC compiler version that is compatible when using C/C++ "back end" code.
    - The year reflects an EasyBuild toolchain, see [FOSS toolchains](https://docs.easybuild.io/common-toolchains/#common_toolchains_overview_foss).



:::{callout} "Some well-known bundles"
:class: dropdown

    - HPC and big data
        - dask
        - mpi4py
        - numba
    - Scientific tools
        - SciPy-bundles: ``numpy``, ``pandas``, ``scipy``
        - xarray
    - Biopython
    - Interactivity
        - iPython
        - JupyterLab
    - Graphics and diagrams
        - Matplotlib
        - Seaborn
    - Machine Learning
        - scikit-learn
        - PyTorch
        - TensorFlow
    - Bundle of useful packages
        - Python-bundle-PyPI

Package     |Bundle module|Also loads        |Avail at *|
------------|-------------|------------------|--------|
numpy       | SciPy-bundle|Python-bundle-PyPI|P, K, C
pandas      | SciPy-bundle|Python-bundle-PyPI|P, K, C
scipy       | SciPy-bundle|Python-bundle-PyPI|P, K, C
matplotlib  | matplotlib  |SciPy-bundle      |P, K, C
seaborn     | Seaborn     |Matplotlib        |P, K, C
biopython   | BioPython   |SciPy-bundle      |P, K, C
dask        | dask        |Matplotlib        |P, K, C
ipython     | IPython     |Python-bundle-PyPI|P, K, C
Jupyterlab  | JupyterLab  |IPython           |P, K, C
xarray      | xarray      |SciPy-bundle      |P, , C
numba       | numba       |SciPy-bundle      |P, K, C
mpi4py      | mpi4py      |OpenMPI           |P, K, C
scikit-learn| scikit-learn|SciPy-bundle      |P, K, C
torch       | PyTorch     |OpenMPI           |P, K, C

* Dardel (``D``), Tetralith (``T``), Alvis (``A``), Pelle (``P``), Kebnekaise (``K``), Cosmos (``C``)

:::{callout} "Some FOSS tool chains and Python version using them"
:class: dropdown

FOSS | Python version| GCC version | Bundle version
-----| --------------|-------------|---------------
2024a| 3.12.3        | 13.3.0      | 2024.06/06
2025b| 3.13.5        | 14.3.0      | 2025.07

- ``foss`` is the full level toolchain.
- ``gfbf`` means that the libraries FlexiBLAS (incl. LAPACK) + FFTW are included.
- ``gompi`` means that the MPI library OpenMPI is included.

- See [Toolchain diagram](https://docs.easybuild.io/common-toolchains/#toolchains_diagram)
:::

:::{danger}

- Make sure to use bundles that are compatible with each-other and with needed Python version.
- Otherwise it is better to create isolated environments with Conda or virtual environments, see [Virtual environments in Python](python_virtual_environments.md).
::: 

:::{callout}  Example Matplotlib
:class: dropdown

- `contourpy`
- `Cycler`
- `fonttools`
- `kiwisolver`
- `matplotlib`

Example Matplotlib 3.10.5-gfbf-2025b

Packages:

- `contourpy-1.3.3`
- `cycler-0.12.1`
- `fonttools-4.58.5`
- `kiwisolver-1.4.8`
- `matplotlib-3.10.5`

Dependencies:

- `Python/3.13.5-GCCcore-14.3.0`
- `Python-bundle-PyPI/2025.07-GCCcore-14.3.0`
- `SciPy-bundle/2025.07-gfbf-2025b`

:::

:::{callout}  What is Python-bundle-PyPI?
:class: dropdown

Bundle of Python packages from PyPI

- Homepage: <https://python.org/>

- Type ``ml help Python-bundle-PyPI/[version]`` on Pelle to see an output.

    Among others:

    - chardet
    - future
    - Jinja
    - pkginfo
    - psutil
    - Pygments
    - pydevtool
    - pytest
    - pytz
    - regex
    - Sphinx
    - threadpoolctl
    - toml
    - urllib

Also loads the package module

    - virtualenv

:::

## Principles with example: ``matplotlib``

- Decide what you need!
    1. Start new project with newest toolchain
    2. Go for version you have used before (reproduce)
    3. Exact versions of many packages may need an isolated environment.
- Load one or several bundles, python is loaded on the fly!

- Check versions

```console
ml spider matplotlib
```

This is a very good way to find packages that are not in a bundle with the same name. For instance, ``pandas`` and ``numpy`` are parts of the ``SciPy-bundle``

or

```console
ml avail matplotlib
```

- Load prerequisites, if needed, and then

```console
ml matplotlib/<version>
```

or

```console
ml matplotlib/<version>
```

- Start Python session in a console with

```console
python
```

- Load a needed library, like

```python
import matplotlib
```

