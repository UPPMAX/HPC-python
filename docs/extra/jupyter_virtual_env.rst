Jupyter in a virtual environment
--------------------------------

.. warning:: 

   **Running Jupyter in a virtual environment**

   You could also use ``jupyter`` (``-lab`` or ``-notebook``) in a virtual environment.

   **UPPMAX**: 

   If you decide to use the --system-site-packages configuration you will get ``jupyter`` from the python module you created your virtual environment with.
   However, you **won't find your locally installed packages** from that jupyter session. To solve this reinstall jupyter within the virtual environment by force::

      $ pip install -I jupyter

   - This overwrites the first version as "seen" by the environment.
   - Then run::

      $ jupyter-notebook
   
   Be sure to start the **kernel with the virtual environment name**, like "Example", and not "Python 3 (ipykernel)".

   **HPC2N**

   To use Jupyter at HPC2N, follow this guide: https://www.hpc2n.umu.se/resources/software/jupyter
   To use it with extra packages, follow this guide after setting it up as in the above guide: https://www.hpc2n.umu.se/resources/software/jupyter-python

