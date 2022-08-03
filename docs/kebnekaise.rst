On Kebnekaise cluster
=====================

There are a few things that are different when using Python on the Kebnekaise cluster compared to the UPPMAX clusters: 

- Anaconda/conda/miniconda are NOT recommended and are not supported. You *can* use it, but we generally ask our users to not install Anaconda on our clusters. We recommand that you consider other options like a virtual environment (or a Singularity container, for the most complicated cases). More information here: https://www.hpc2n.umu.se/documentation/guides/anaconda 
- When loading Python or Python packages, there are prerequisites that needs to be loaded first. See which ones with ``module spider <Python>/<version>`` or ``module spider <Python-package>/<version>`` 
- We have many Python packages site-installed as modules. Some of these are

  - ASE
  - Keras
  - PyTorch
  - SciPy-bundle (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)
  - TensorFlow
  - Theano
  - matplotlib
  - scikit-learn
  - scikit-image
  - pip
  - iPython
  - Cython
  
- We do not currently have Jupyter installed, but there are ways for users to run it. Since it is somewhat involved, please contact us at support@hpc2n.umu.se for more information

Accessing Kebnekaise
--------------------

**Kebnekaise login node**

Login with 

.. code-block:: sh

    ssh <hpc2n-username>@kebnekaise.hpc2n.umu.se
    
using your favourite SSH client. More information about this here: https://www.hpc2n.umu.se/access/login 

**Kebnekaise ThinLinc login node**

If you do not have a preferred SSH client installed, then this is the recommended way to login, as it comes with a GUI environment directly and no need to run an X11 server. 

Use 

.. code-block:: sh

    kebnekaise-tl.hpc2n.umu.se
    
as the server in the ThinLinc login. 

There is a guide for you to follow here: https://www.hpc2n.umu.se/documentation/guides/thinlinc 
