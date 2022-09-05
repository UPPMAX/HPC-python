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
    
**Make an overview general for both clusters**
- general procedure
- the tools
   - venv
   - *virtualenv*
   - Conda
- point to separated sessions

.. admonition:: venv vs. virtualenv

   - These are almost completely interchangeable
   - the difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.


Virtual environment - venv (UPPMAX)
-----------------------------------

Seperatee session at <https://uppmax.github.io/HPC-python/isolatedUPPMAX.html>

Virtual environment - virtualenv (HPC2N)
----------------------------------------

Seperate session at <https://uppmax.github.io/HPC-python/isolatedHPC2N.html>

.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environemnts.
      - UPPMAX has  Conda and venv
      - HPC2N has virtualenv.
     -  More details in the seperated sessions!
 
   
