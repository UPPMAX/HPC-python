Isolated environments
=====================

.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments", as well as installing using setup.py! Installing with a virtual environment is the only recommended way at HPC2N! 

.. questions::

   - How to work with isolated environments at HPC2N and UPPMAX?
   - How do you structure a lesson effectively for teaching?

   
.. objectives:: 

   - Give a general 'theoretical* introduction to isolated environments 
   - Site-specific procedures are given at the separated sessions.

General procedures   
------------------

You will often have the situation that your project(s) use different versions of Python and different versions of packages. This is great if you need different versions of a package for different tasks, for instance.
As an example, maybe you have been using TensorFlow 1.x.x for your project and now you need to install a package that requires TensorFlow 2.x.x but you will still be needing the old version of TensorFlow for another package, for instance. This is easily solved with isolated environments.

Isolated environments lets you create separate workspaces for different versions of Python and/or different versions of packages. You can activate and deactivate them one at a time, and work as if the other workspace does not exist.

There are different tools for creating an isolated environement, but they all have some things in common. At both UPPMAX and HPC2N the workflow is: 

- You load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option)
- You create the isolated environment with something like venv, virtualenv, or conda
- You activate the environment
- You install (or update) the environment with the packages you need
- You work in the isolated environment
- You deactivate the environment after use 

**The tools**

In this course we will look at the following tools for creating and using isolated environments: 

   - venv   
   - *virtualenv*
   - Conda


.. admonition:: venv vs. virtualenv

   - These are almost completely interchangeable
   - the difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.


Virtual environment - venv (UPPMAX)
-----------------------------------

Separate session at <https://uppmax.github.io/HPC-python/isolatedUPPMAX.html>

Virtual environment - virtualenv (HPC2N)
----------------------------------------

Separate session at <https://uppmax.github.io/HPC-python/isolatedHPC2N.html>

.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environemnts.
      - UPPMAX has  Conda and venv
      - HPC2N has virtualenv.
      - More details in the separated sessions!
 
   
