Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners 

   - can decide on useful file formats
   - can allocate resources sufficient to data size
   - can use data-chunking as technique
   - know where to learn more
    
Why it matters
---

scenario
::::::::

- use dataset (10 GB)
- fails in pandas or is slow
- Load with dask + xarray

What the constraints are
------------------------

Memory, nodes

Solutions and tools
-------------------

File formats
::::::::::::

RAM
:::

- Mention memory per core considerations.
- Show SLURM options for memory and time.
- Briefly explain what happens when a Dask job runs on multiple cores.

!!! info "Keywords"

    OOM

Chunking
::::::::

Tools like Dask and xarray handle chunking 
automatically. Add one short diagram showing:
Big file → split into chunks → parallel workers → results combined.

!!! info "Keywords"

    - chunk size
    - lazy execution
    - meta-data-rich arrays

!!! note "Sum up"

    - Load Python modules and activate virtual environments.
    - Request appropriate memory and runtime in SLURM.
    - Store temporary data in local scratch ($SNIC_TMP).
    - Check job memory usage with sacct or sstat.

Summary
-------

Data source → Format choice → Load/Chunk → Process → Write

Exercises
---------


- Pandas 
- xarray
- dask





