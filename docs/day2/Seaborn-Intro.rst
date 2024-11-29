################################################################
A Brief Introduction to the Seaborn Statistical Plotting Library
################################################################

Seaborn is a plotting library built predominantly on Matplotlib and designed to quickly and easily create presentation-ready statistical plots from Pandas data structures.

.. tabs:: Loading Seaborn

  .. tab:: HPC2N

     As usual, you can check ``ml spider Seaborn`` to see the available versions and how to load them. These Seaborn modules are built to load their Matplotlib and SciPy-bundle dependencies internally.

     If you work at the command line, after importing Matplotlib, you will need to set ``matplotlib.use('Tkinter')`` in order to view your plots. This is not necessary if you work in a GUI like Jupyter or Spyder.

     As of 27-11-2024, ``ml spider Seaborn`` outputs the following versions on Kebnekaise:

     .. code-block:: console

        ----------------------------------------------------------------------------
          Seaborn:
        ----------------------------------------------------------------------------
            Description:
              Seaborn is a Python visualization library based on matplotlib. It
              provides a high-level interface for drawing attractive statistical
              graphics. 
        
             Versions:
                Seaborn/0.12.1
                Seaborn/0.12.2
                Seaborn/0.13.2
        
        ----------------------------------------------------------------------------
          For detailed information about a specific "Seaborn" package (including how to load the modules) use the module's full name.
          Note that names that have a trailing (E) are extensions provided by other modules.
          For example:
        
             $ module spider Seaborn/0.13.2
        ----------------------------------------------------------------------------

  .. tab:: LUNARC
  
       On COSMOS, it is recommended that you use the On-Demand Spyder or Jupyter Lab applications to use Seaborn. These applications are configured to load Seaborn and all its dependencies autonatically, including the SciPy-bundle. The demonstrations will be done on Cosmos with Spyder.
  
       If you must work on the command line, then you will need to load Seaborn separately, along with any prerequisite modules. After importing Matplotlib, you will need to set ``matplotlib.use('Tkinter')`` in order to view your plots.
  
       As of 27-11-2024, ``ml spider Seaborn`` outputs the following versions on COSMOS:
  
       .. code-block:: console

          ----------------------------------------------------------------------------
            Seaborn:
          ----------------------------------------------------------------------------
              Description:
                Seaborn is a Python visualization library based on matplotlib. It
                provides a high-level interface for drawing attractive statistical
                graphics. 
          
               Versions:
                  Seaborn/0.11.2
                  Seaborn/0.12.1
                  Seaborn/0.12.2
                  Seaborn/0.13.2
          
          ----------------------------------------------------------------------------
            For detailed information about a specific "Seaborn" package (including how to 
          load the modules) use the module's full name.
            Note that names that have a trailing (E) are extensions provided by other modu
          les.
            For example:
          
               $ module spider Seaborn/0.13.2
          ----------------------------------------------------------------------------

  .. tab:: UPPMAX

     On Rackham, Seaborn/0.13.2 is included in ``python_ML_packages/3.11.8-cpu``. Jupyter-Lab is available but Spyder is not installed centrally.

Seaborn can produce a wide variety of statistical plots, and even offers built-in regression functions, 
but in the interest of time, we will focus on just a few that are hard to replicate with Matplotlib 
alone: ``catplot()``, ``jointplot()``, ``heatmap()``, and ``pairplot()`` (or ``PairGrid()``).

.. caution:: Don't Rely on Seaborn for Regression Analysis

  We will not cover regression because Seaborn does not return any parameters needed to assess the 
  quality of the fit. The official documentation itself warns that the Seaborn regression functions are
  only intended for quick and dirty visualizations to *motivate* proper in-depth analysis.



