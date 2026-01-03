Summary day 2
=============

- `Summary of third  day <./summary3.html>`_
- `Summary of fourth day <./summary4.html>`_


.. keypoints::

   - Load and run and pre-installed packages
      - Use python from module system
      - Start a Python shell session either with ``python`` or ``ipython``
      - run scripts with ``python3 <script.py>``
      - Check for preinstalled packages
         - from the Python shell with the ``import`` command
         - from BASH shell with the
            - ``pip list`` command at both centers
            - ``ml help python/3.11.8`` at UPPMAX
            - ``module -r spider '.*Python.*'`` at otherwise
     
   - Install packages and use isolated environments 
      - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
      - Make it for each project you have for reproducibility.
      - There are different tools to create virtual environments.
         - ``virtualenv`` and ``venv``
            - install packages with ``pip``.
            - the flag ``--system-site-packages`` includes preinstalled packages as well
      - Conda (available everywhere but not HPC2N)
         - Conda is an installer of packages but also bigger toolkits
             - Conda creates isolated environments as well
             - requires that you install all packages needed. 
         - Rackham: Pip or secondary conda
         - Bianca: conda and secondary wharf + (pip or conda)

   - Interactive work on calculation nodes
      - Start an interactive session on a calculation node by a SLURM allocation (similar flags)
         - At HPC2N: ``salloc`` ...
         - At UPPMAX/NSC: ``interactive`` ...
         - At LUNARC: Desktop on demand
      - Follow the same procedure as usual by loading the Python module and possible prerequisites.

   - IDEs
       - Jupyter-lab/notebook
           - Available in all clusters
           - OnDemand at Dardel, Alvis, Cosmos and Kebnekaise
       - Spyder
           - Best available at LUNARC (from OnDemand as well)
           - Possible at the others through virtual environments (pip) or Conda (not HPC2N)
       - VScode
           - Always available (except for Bianca) from local computer (if you have VS Code)
           - available as modules or 
           - from onDemand at Cosmos and Kebnekaise

