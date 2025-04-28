Summary day 3
=============

`Summary of first day <./summary1.html>`_
`Summary of first day <./summary2.html>`_
`Summary of first day <./summary4.html>`_

.. keypoints::

   - Intro to Pandas

       - Lets you construct list- or table-like data structures with mixed data types, the contents of which can be indexed by arbitrary row and column labels
       - The main data structures are Series (1D) and DataFrames (2D). Each column of a DataFrame is a Series

   - Seaborn
       - Seaborn makes statistical plots easy and good-looking!

       - Seaborn plotting functions take in a Pandas DataFrame, sometimes the names of variables in the DataFrame to extract as x and y, and often a hue that makes different subsets of the data appear in different colors depending on the value of the given categorical variable.

   - Batch mode
      - The SLURM scheduler handles allocations to the calculation nodes
      - Batch jobs runs without interaction with user
      - A batch script consists of a part with *SLURM parameters* describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
      - Remember to include possible input arguments to the Python script in the batch script.
   
   - Big data

       - allocate resources sufficient to data size
       - decide on useful file formats
       - use data-chunking as technique
