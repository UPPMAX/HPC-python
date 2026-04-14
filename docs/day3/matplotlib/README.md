
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(matplotlib)=

# `matplotlib`

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- understand why `matplotlib` is important
- have run Python code that uses `matplotlib`

:::

## Loading Matplotlib

- Use the documentation of the HPC cluster you work on

:::{admonition} Answer: where is your documentation?
:class: dropdown

Sorted by HPC cluster:

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->
HPC center |HPC cluster|HPC cluster-specific documentation
-----------|-----------|------------------------------------------------------------
C3SE       |Alvis      |[Documentation](https://www.c3se.chalmers.se)
UPPMAX     |Bianca     |[Documentation](https://docs.uppmax.uu.se)
LUNARC     |COSMOS     |[Documentation](https://lunarc-documentation.readthedocs.io)
PDC        |Dardel     |[Documentation](https://support.pdc.kth.se)
HPC2N      |Kebnekaise |[Documentation](https://docs.hpc2n.umu.se)
UPPMAX     |Pelle      |[Documentation](https://docs.uppmax.uu.se)
NSC        |Tetralith  |[Documentation](https://www.nsc.liu.se)
<!-- markdownlint-enable MD013 -->
:::

- In that documentation, find the software module to load the package.
  If you know how, you may also use the module system

:::{admonition} Answer: where is the `matplotlib` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `matplotlib` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |[`matplotlib` documentation](https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy)
Bianca     |[`matplotlib` documentation](https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu)
COSMOS     |[`matplotlib` documentation](https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python)
Dardel     |[`matplotlib` documentation](https://support.pdc.kth.se/doc/applications/tensorflow) (irrelevant)
Kebnekaise |[`matplotlib` documentation](https://docs.hpc2n.umu.se/software/apps/#scipy)
Pelle      |[`matplotlib` documentation](https://docs.uppmax.uu.se/software/python_bundles/#pytorch)
Tetralith  |[`matplotlib` documentation](https://www.nsc.liu.se/software/python)

<!-- markdownlint-enable MD013 -->

:::

- Load the software module to use `matplotlib`

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Matplotlib
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |`module load matplotlib/3.9.2-gfbf-2024a`
COSMOS     |`module load matplotlib/3.8.2` (avoid version `3.9.2`!)
Dardel     |`module load PDC/23.12 cray-python/3.11.5 matplotlib/3.8.2-cpeGNU-23.12`
Kebnekaise |`module load matplotlib/3.8.2`
Pelle      |`module load matplotlib/3.9.2-gfbf-2024a`
Tetralith  |`module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2`

<!-- markdownlint-enable MD013 -->

## Exercises

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# plt.show()
plt.figure().savefig('my_figure.png') # Unsure if this works

```

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
