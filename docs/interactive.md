# Interactive work on the compute nodes

There are several ways to run Python interactively

- Directly on the login nodes: **only** do this for short jobs that do not take a lot of resources
- As an interactive job on the computer nodes, launched via the batch system
- Jupyter notebooks (UPPMAX suggests installing your own version with conda) 

UPPMAX
------

HPC2N
-----

Python on the login nodes
+++++++++++++++++++++++++

It is possible to run Python directly on the Kebnekaise login node or the Kebnekaide ThinLinc login node. 

**NOTE** This should *only* be done for short jobs or jobs that do not use a lot of resources, as the login nodes can otherwise become slow for all users. 

**Kebnekaise login node**

Login with 

.. code-block:: sh

    ssh <hpc2n-username>@kebnekaise.hpc2n.umu.se
    
using your favourite SSH client. More information about this here: https://www.hpc2n.umu.se/access/login 

**Kebnekaise ThinLinc login node**

If you do not have a preferred SSH client installed, then this is the recommended way to login, as it comes with a GUI environment directly and no need to run an X11 server. 

Use 

.. code-block:: sh

    kebnekaise-tl-hpc2n.umu.se
    
as the server in the ThinLinc login. 

There is a guide for you to follow here: https://www.hpc2n.umu.se/documentation/guides/thinlinc 

**Running Python**

As mentioned under the `Load and run python <https://uppmax.github.io/HPC-python/load_run.html>`_ section, you first need to load Python and its prerequisites, then any modules you need, then activate any virtual environment you have installed Python packages to. Then start Python. So, the way to do this is: 

1) Load Python and prerequisites: `module load <pre-reqs> Python/<version>``
2) Load site-installed Python packages (optional): ``module load <pre-reqs> <python-package>/<version>``
3) Activate your virtual environment (optional): source 

.. admonition:: Example, Python 3.9.5, site-installed numpy, and own-installed spacy
    :class: dropdown
   
        .. code-block:: sh
