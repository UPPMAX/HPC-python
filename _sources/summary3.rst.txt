Summary day 3
=============

- `Summary of second day <./summary2.html>`_
- `Summary of fourth day <./summary4.html>`_

.. keypoints::

   - Intro to matplotlib
       - Matplotlib is the essential Python data visualization package, with nearly 40 different plot types to choose from depending on the shape of your data and which qualities you want to highlight.
       - Almost every plot will start by instantiating the figure, ``fig`` (the blank canvas), and 1 or more axes objects, ``ax``, with ``fig, ax = plt.subplots(*args, **kwargs)``.
       - There are several ways to tile subplots depending on how many there are, how they are shaped, and whether they require non-Cartesian coordinate systems.
       - Most of the plotting and formatting commands you will use are methods of ``Axes`` objects. (A few, like ``colorbar`` are methods of the ``Figure``, and some commands are methods both.)
   - Intro to Pandas
       - Lets you construct list- or table-like data structures with mixed data types, the contents of which can be indexed by arbitrary row and column labels
       - The main data structures are Series (1D) and DataFrames (2D). Each column of a DataFrame is a Series

   - Seaborn
       - Seaborn makes statistical plots easy and good-looking!

       - Seaborn plotting functions take in a Pandas DataFrame, sometimes the names of variables in the DataFrame to extract as x and y, and often a hue that makes different subsets of the data appear in different colors depending on the value of the given categorical variable.

   - Big data
       - Allocate more RAM by asking for

           - Several cores
           - Nodes will more RAM
           - Check job memory usage with ``sacct`` or ``sstat``. Check you documentation!
       - File formats

           - No format fits all requirements
           - HDF5 and NetCDF good for Big data since it allows loading parts of the file into memory
       - Store temporary data in local scratch ($SNIC_TMP).
       - Packages

           - xarray

               - can deal with 3D-data and higher dimensions
           - Dask

               - uses lazy execution
               - Only use for processing very large amount of data
               - Chunking: Data source → Format choice → Load/Chunk → Process → Write

   - Batch mode
       - The SLURM scheduler handles allocations to the calculation nodes
       - Batch jobs runs without interaction with user
       - A batch script consists of a part with *SLURM parameters* describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
       - Remember to include possible input arguments to the Python script in the batch script.
   
