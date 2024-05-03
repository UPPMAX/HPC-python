# Directory overview for programs folder

Contains Python scripts, as well as a few Julia and Fortran 90 scripts. Also contains some data files needed for the Python scripts. 

## Python scripts and data files 

In the below, U is UPPMAX, H is HPC2N, R is Rackham, S is Snowy, and K is Kebnekaise. 

| Name | Section(s) used | Modules needed | System | Related batch script | Comments |  
| ---- | --------------- | -------------- | ------ | -------------------- | -------- |
| example.py | Load/run | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 | U, H | None | |
| pandas_matplotlib-kebnekaise.py | Load/run | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-kebnekaise.py | Load/run | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-rackham.py | Load/run | python/3.11.8 | U (R) | None | Will be modified in the ML section |
| pandas_matplotlib-linreg-rackham.py | Load/run | python/3.11.8 | U(R) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-pretty-kebnekaise.py | Load/run | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | (H(K) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-pretty-rackham.py | Load/run | python/3.11.8 | U(R) | None | Will be modified in the ML section | 

| ---- | --------------- | -------------- | ------ | -------------------- | -------- |
| mmmult.py | Batch | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 | U, H | run_mmmult.sh (U, H) | | 
| compute.py | Batch | U: uppmax python/3.11.8 python_ML_packages/3.11.8-gpu <br> H: GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 SciPy-bundle/2023.07 numba/0.58.1  | U(S), H(K) | run_compute.sh (U, H) | | 
| sum-2args.py | Batch | U: python/3.11.8 <br> H: GCC/12.3.0 Python/3.11.3 | U(R), H(K) | run_sum-2args.sh (U, H) | | 

| ---- | --------------- | -------------- | ------ | -------------------- | -------- |
| add2.py | Interactive | U: python/3.11.8 <br> H: GCC/12.3.0 Python/3.11.3 | U(R), H(K) | None | | 
| ---- | --------------- | -------------- | ------ | -------------------- | -------- | 
| integration2d_serial.py | Parallel | U: python/3.9.5 <br> H: GCCcore/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| integration2d_serial_numba.py | Parallel | U: python/3.9.5 <br> H: GCCcore/11.2.0 Python/3.9.6 | U(S), H(K) | | Make a virtual environment, activate it, and install numba with pip before running. |  
| call_fortran_code.py | Parallel | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | Run in same virtual environment as you compiled ``fortran_function.f90`` in | 
| call_julia_code.py | Parallel | U: python/3.9.5 julia/1.7.2 <br> H: GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 Julia/1.9.3-linux-x86_64 | U, H | None | After doing as mentioned under ``julia_function.jl``, run with ``python call_julia_code.py`` | 
| integration2d_threading.py | Parallel | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| dot.py | Parallel | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | U(R), H(K) | None | Activate the same environment that numpy is installed in, as you used for ``fortran_function.f90``. Do ``export OMP_NUM_THREADS=<numthreads>``, then run with ``python dot.py``. Try several values of ``numthreads``. |  
| call_fortran_code_openmp.py | Parallel | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | Run in same virtual environment as you compiled ``fortran_function_openmp.f90`` in. Set ``export OMP_NUM_THREADS=4`` first. | 
| integration2d_multiprocessing.py | Parallel | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| integration2d_mpi.py | Parallel |   
- add-list.py                       
- call_fortran_code_openmp.py       
- example-tf.py                     
- hello-world-array.py             
- integration2d_gpu.py      
- integration2d_gpu_shared.py       
- pandas_matplotlib-batch.py
- pandas_matplotlib-linreg-batch.py
- pandas_matplotlib-linreg-pretty-batch.py
- pytorch_fitting_gpu.py
- seaborn-example.py
- simple_example.py
- Transfer_Learning_NLP_Horovod.py

### Data files related to above Python scripts

| Name | Section(s) used | Related Python scripts | Related batch scripts | Comments | 
| ---- | --------------- | ---------------------- | --------------------- | -------- | 
| mtcars.csv | 
| regression.test | 
| regression.train | 
| scottish_hills.csv | 

## Julia scripts 

| Name | Section(s) used | Related Python scripts | Related batch scripts | Modules | Comments |
| ---- | --------------- | ---------------------- | --------------------- | ------- | -------- |
| julia_function.jl | Parallel | call_julia_code.py | None | U: python/3.9.5 julia/1.7.2 <br> H: GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 Julia/1.9.3-linux-x86_64 | H/U: install julia with pip in the same virtual environment as you installed numba in for ``integration2d_serial_numba.py``, as we need PyJulia. Will be run from ``call_julia_code.py``. Test by starting ``python`` and doing ``import julia``. | 

## Other

| Name | Section(s) used | Related Python scripts | Related batch scripts | Modules | Comments |
| ---- | --------------- | ---------------------- | --------------------- | ------- | -------- | 
| fortran_function.f90 | Parallel | call_fortran_code.py | None | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | H/U: install numpy with pip (in the same virtual environment as you installed numba for ``integration2d_serial_numba.py``). <br> H/U: compile with ``f2py -c -m myfunction fortran_function.f90`` | 
| fortran_function_openmp.f90 | Parallel | call_fortran_code_openmp.py | None | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | H/U: use same virtual environment as you installed numpy in for ``fortran_function.f90``). <br> H/U: compile with ``f2py -c --f90flags='-fopenmp' -lgomp -m myfunction_openmp fortran_function_openmp.f90`` |

