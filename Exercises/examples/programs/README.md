# Directory overview for programs folder

Contains Python scripts, as well as a few Julia and Fortran 90 scripts. Also contains some data files needed for the Python scripts. 

## Python scripts and data files 

In the below, U is UPPMAX, H is HPC2N, R is Rackham, S is Snowy, and K is Kebnekaise. Batch scripts for HPC2N are located in the ``/Exercises/examples/hpc2n/`` directory and the ones for UPPMAX are located in ``Exercises/examples/uppmax``. 

### Section "Loading and running"

| Name | Modules needed | System | Related batch script | Comments |  
| ---- | -------------- | ------ | -------------------- | -------- |
| example.py | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 | U, H | None | |
| pandas_matplotlib-kebnekaise.py | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-kebnekaise.py | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H (K) | None | Will be modified in the ML section | 
| pandas_matplotlib-rackham.py | python/3.11.8 | U (R) | None | Will be modified in the ML section |
| pandas_matplotlib-linreg-rackham.py | python/3.11.8 | U(R) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-pretty-kebnekaise.py | GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | H(K) | None | Will be modified in the ML section | 
| pandas_matplotlib-linreg-pretty-rackham.py | python/3.11.8 | U(R) | None | Will be modified in the ML section | 

### Section "Batch" 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- |
| mmmult.py | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 | U, H | run_mmmult.sh (U, H) | |
| hello-world-array.py | U: python/3.11.8 <br>H: GCC/12.3.0 Python/3.11.3 | U, H | hello-world-array.sh (U, H) | |  
| compute.py | U: uppmax python/3.11.8 python_ML_packages/3.11.8-gpu <br> H: GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 SciPy-bundle/2023.07 numba/0.58.1  | U(S), H(K) | run_compute.sh (U, H) | | 
| sum-2args.py | U: python/3.11.8 <br> H: GCC/12.3.0 Python/3.11.3 | U(R), H(K) | run_sum-2args.sh (U, H) | | 

### Section "Interactive" 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- |
| add2.py | U: python/3.11.8 <br> H: GCC/12.3.0 Python/3.11.3 | U(R), H(K) | None | | 

### Section "Parallel" 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- | 
| integration2d_serial.py | U: python/3.9.5 <br> H: GCCcore/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| integration2d_serial_numba.py | U: python/3.9.5 <br> H: GCCcore/11.2.0 Python/3.9.6 | U(S), H(K) | | Make a virtual environment, activate it, and install numba with pip before running. |  
| call_fortran_code.py | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | Run in same virtual environment as you compiled ``fortran_function.f90`` in | 
| call_julia_code.py | U: python/3.9.5 julia/1.7.2 <br> H: GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 Julia/1.9.3-linux-x86_64 | U, H | None | After doing as mentioned under ``julia_function.jl``, run with ``python call_julia_code.py`` | 
| integration2d_threading.py | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| dot.py | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | U(R), H(K) | None | Activate the same environment that numpy is installed in, as you used for ``fortran_function.f90``. Do ``export OMP_NUM_THREADS=<numthreads>``, then run with ``python dot.py``. Try several values of ``numthreads``. |  
| call_fortran_code_openmp.py | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | Run in same virtual environment as you compiled ``fortran_function_openmp.f90`` in. Set ``export OMP_NUM_THREADS=4`` first. | 
| integration2d_multiprocessing.py | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 | U(R), H(K) | None | | 
| integration2d_mpi.py | U: python/3.9.5 gcc/9.3.0 openmpi/3.1.5 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | U, H | integration2d_mpi.sh (U, H) | Create a virtual environment, activate it, and install mpi4py in it | 

### Section "GPU" 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- |
| add-list.py | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 CUDA/11.4.1 | add-list.sh (U, H) | First create a virtual environment, activate it, and install numba with pip. | 
| integration2d_gpu.py <br> integration2d_gpu_shared.py | U: python/3.9.5 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 CUDA/11.4.1 | job-gpu.sh (H) or integration2d_gpu_shared.sh (U, H) | You need to use the same virtual environment you created for ``add-list.py`` | 

### Section "ML" 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- |
| pandas_matplotlib-batch-<kebnekaise/rackham>.py | U: python/3.11.8 <br> H: GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | U(R), H(K) | run_pandas_matplotlib-batch.sh (U, H) | |  
| pytorch_fitting_gpu.py | U: uppmax python/3.11.8 python_ML_packages/3.11.8-gpu <br> H: GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 | U(S), H(K) | pytorch_fitting_gpu.sh (U(S), H(K)) | |
| example-tf.py | U: uppmax python_ML_packages/3.11.8-gpu <br> H: GCC/11.3.0 Python/3.10.4 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2 | U(S), H(K) | example-tf.sh (U, H) | | 
| pandas_matplotlib-linreg-batch-<kebnekaise/rackham>.py | U: python/3.11.8 <br> GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | U(R), H (K) | pandas_matplotlib-linreg-batch.sh | |
| pandas_matplotlib-linreg-pretty-batch-<kebnekaise/rackham>.py | U: python/3.11.8 <br> GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | U(R), H (K) | pandas_matplotlib-linreg-pretty-batch.sh | |
| simple-lightgbm.py | U: uppmax <br> H: GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 | U, H | simple-lightgbm.sh (U, H) | You need a virtual environment with lightgbm (and scipy for HPC2N) installed and activated to run this | 

### Extra/other 

| Name | Modules needed | System | Related batch script | Comments |
| ---- | -------------- | ------ | -------------------- | -------- |
| Transfer_Learning_NLP_Horovod.py | U: uppmax python_ML_packages python/3.9.5 gcc/10.3.0 build-tools cmake/3.22.2 <br> H: GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 TensorFlow/2.4.1 Horovod/0.21.1-TensorFlow-2.4.1 | U, H | run_horovod.sh (U, H) | A virtual environment with tensorflow_hub and sklearn installed with pip is needed on Kebnekaise, and one with horovod and tensorflow-hub installed with pip is needed on Rackham/Snowy. | 
| seaborn-example.py | | | None | Old example. Needed seaborn installed in a virtual environment |  

### Data files related to above Python scripts

| Name | Section(s) used | Related Python scripts | Related batch scripts | Comments | 
| ---- | --------------- | ---------------------- | --------------------- | -------- | 
| mtcars.csv | None (old example) | seaborn-example.py | None | Old example. Needed seaborn installed in a virtual environment | 
| regression.test | ML | simple_lightgbm.py | simple-lightgbm.sh (U, H) | | 
| regression.train | ML | simple_lightgbm.py | simple-lightgbm.sh (U, H) | | 
| scottish_hills.csv | Load/run, ML | pandas_matplotlib-*.py | pandas_matplotlib-*.sh | | 

## Julia scripts 

| Name | Section(s) used | Related Python scripts | Related batch scripts | Modules | Comments |
| ---- | --------------- | ---------------------- | --------------------- | ------- | -------- |
| julia_function.jl | Parallel | call_julia_code.py | None | U: python/3.9.5 julia/1.7.2 <br> H: GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 Julia/1.9.3-linux-x86_64 | H/U: install julia with pip in the same virtual environment as you installed numba in for ``integration2d_serial_numba.py``, as we need PyJulia. Will be run from ``call_julia_code.py``. Test by starting ``python`` and doing ``import julia``. | 

## Other

| Name | Section(s) used | Related Python scripts | Related batch scripts | Modules | Comments |
| ---- | --------------- | ---------------------- | --------------------- | ------- | -------- | 
| fortran_function.f90 | Parallel | call_fortran_code.py | None | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | H/U: install numpy with pip (in the same virtual environment as you installed numba for ``integration2d_serial_numba.py``). <br> H/U: compile with ``f2py -c -m myfunction fortran_function.f90`` | 
| fortran_function_openmp.f90 | Parallel | call_fortran_code_openmp.py | None | U: python/3.9.5 gcc/10.3.0 <br> H: GCC/11.2.0 Python/3.9.6 OpenMPI/4.1.1 | H/U: use same virtual environment as you installed numpy in for ``fortran_function.f90``). <br> H/U: compile with ``f2py -c --f90flags='-fopenmp' -lgomp -m myfunction_openmp fortran_function_openmp.f90`` |

