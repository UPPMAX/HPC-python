################################################################
A Brief Introduction to the Seaborn Statistical Plotting Library
################################################################

Seaborn is a plotting library built entirely with Matplotlib and designed to quickly and easily create presentation-ready statistical plots from Pandas data structures.

Seaborn can produce a wide variety of statistical plots, and even offers built-in regression functions, but in the interest of time, we will focus on just a few that are hard to replicate with Matplotlib alone: ``catplot()``, ``jointplot()``, ``heatmap()``, and ``pairplot()`` (or ``PairGrid()``). We will make extensive use of Seaborn's built-in testing datasets, of which there are many. You will have seen a couple already in the Matplotlib and Pandas lectures.

.. caution:: Don't Rely on Seaborn for Regression Analysis

   We will not cover regression because Seaborn does not return any parameters needed to assess the 
   quality of the fit. The official documentation itself warns that the Seaborn regression functions are
   only intended for quick and dirty visualizations to *motivate* proper in-depth analysis.


Load and Run Seaborn
--------------------

.. tabs::

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

In all cases, once Seaborn or the module that provides it is loaded, it can be imported directly in Python. The typical abbreviation in online documentation is ``sns``, but for those of us who never watched The West Wing, ``sb`` is fine and is what will be used in this tutorial.


Common Features
---------------

Sample Datasets
^^^^^^^^^^^^^^^^

This tutorial will make use of some of the free test data sets that Seaborn provides with the ``.load_dataset()`` function. These are also handy for playing with Pandas and a variety of machine learning packages (TensorFlow, PyTorch, etc.). The full list of datasets can be viewed with ``sb.get_dataset_names()``. A few of the more popular data sets include...

* ``'penguins'``, sex-segregated measurements of the beaks, flippers, and body masses of 3 species of penguins that live on the Antarctic Peninsula.
* ``'iris'``, measurements of the petal and sepal dimensions of three species of iris flower.
* ``'titanic'``, records of the ticket class, demographics, and survival status of passengers on the Titanic
* ``mpg``, information about the make, model, physical characteristics, engine specifications, and fuel economy of a variety of cars.

We will use some of these datasets in this tutorial.

Commonalities in Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^

Seaborn plotting functions are designed to take Pandas DataFrames (or sometimes Series) as inputs. As such, different plot types share many of the same kwargs (there are no args). The following are the most important:

* ``data``---the DataFrame in which to search for the remaining kwargs. This kwarg is mandatory.
* ``x`` and ``y``---the names of two columns in your DataFrame to plot against each other. These are usually necessary, but not if you're plotting every possible pairing of numerical data columns against each other all at once, as in ``PairGrid`` or ``heatmap``.
* ``hue``---this kwarg accepts a categorical variable (e.g. species, sex, brand, etc.) column name, groups the data by those categories, and plots them all on the same plot in a different color.

Apart from the actual plotting commands, much of the customization of the figure and axes is still done in ``matplotlib``, so you typically want to import that as well.

.. caution::

   Seaborn typically titles axes by the variable names as they appear in the DataFrame, underscores and all. These titles can be tedious to correct with proper typesetting. Since proper typesetting is necessary for figures to be published, an upcoming example will demonstrate how to fix the axis label formatting.


Plotting with Seaborn
---------------------

Here we will explore a few of the plot types Seaborn offers that are difficult to replicate in Matplotlib.
