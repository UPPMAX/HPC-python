# Directory overview for programs folder

Contains Python scripts, as well as a few Julia and Fortran 90 scripts. Also contains some data files needed for the Python scripts. 

## Python scripts and data files 

In the below, U is UPPMAX, H is HPC2N, R is Rackham, S is Snowy, and K is Kebnekaise. 

| Name | Section used | Modules needed | System | Related batch script | Comments |  
| ---- | ------------ | -------------- | ------ | -------------------- | -------- |
| example.py | Load/run | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 | U, H | None | |
| pandas_matplotlib-kebnekaise.py | Load/run | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-kebnekaise.py | Load/run | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-rackham.py | Load/run | python/3.11.8 | U (R) | Will be modified in the ML section |
pandas_matplotlib-linreg-rackham.py

- add2.py                           
- add-list.py                       
- call_fortran_code_openmp.py       
- call_fortran_code.py              
- call_julia_code.py               
- compute.py                        
- dot.py                           
- example-tf.py                     
- hello-world-array.py             
- integration2d_gpu.py      
- integration2d_gpu_shared.py       
- integration2d_mpi.py             
- integration2d_multiprocessing.py  
- integration2d_serial_numba.py     
- integration2d_serial.py          
- integration2d_threading.py        
- mmmult.py             
- pandas_matplotlib-batch.py
- pandas_matplotlib-linreg-batch.py
- pandas_matplotlib-linreg-pretty-batch.py
- pandas_matplotlib-linreg-pretty-kebnekaise.py
- pandas_matplotlib-linreg-pretty-rackham.py
- pytorch_fitting_gpu.py
- seaborn-example.py
- simple_example.py
- sum-2args.py
- Transfer_Learning_NLP_Horovod.py

### Data files related to above Python scripts

- mtcars.csv
- regression.test
- regression.train
- scottish_hills.csv

## Julia scripts 

- julia_function.jl

## Other

- fortran_function.f90
- fortran_function_openmp.f90

