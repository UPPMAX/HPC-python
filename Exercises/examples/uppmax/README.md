# Directory overview for uppmax folder

Contains batch scripts for running on UPPMAX's clusters Rackham and Snowy. 

In the below, U is UPPMAX, H is HPC2N, R is Rackham, S is Snowy, and K is Kebnekaise. Python scripts are located in the ``/Exercises/examples/programs/`` directory.  

## Section "Batch" 

| Name | System | Related Python script | Comments |
| ---- | ------ | --------------------- | -------- |
| run_mmmult.sh | U, H | mmmult.py | |
| hello-world-array.sh | U, H | hello-world-array.py | |  
| run_compute.sh | U, H | compute.py | | 
| run_sum-2args.sh | U, H | sum-2args.py | | 

## Section "Parallel" 

| Name | System | Related Python script | Comments |
| ---- | ------ | --------------------- | -------- |
| integration2d_mpi.py | U, H | integration2d_mpi.sh | | 

## Section "GPU" 

| Name | System | Related Python script | Comments |
| ---- | ------ | --------------------- | -------- |
| add-list.sh | U, H | add-list.sh | | 
| job-gpu.sh | H | integration2d_gpu.py <br> integration2d_gpu_shared.py | | 
| integration2d_gpu_shared.sh | U, H | integration2d_gpu.py <br> integration2d_gpu_shared.py | | 

## Section "ML" 

| Name | System | Related Python script | Comments |
| ---- | ------ | --------------------- | -------- |
| run_pandas_matplotlib-batch.sh | U(R), H(K) | pandas_matplotlib-batch-<kebnekaise/rackham>.py | |  
| pytorch_fitting_gpu.sh | U(S), H(K) | pytorch_fitting_gpu.py | |
| example-tf.sh | U(S), H(K) | example-tf.py | | 
| pandas_matplotlib-linreg-batch.sh | U(R), H (K) | pandas_matplotlib-linreg-batch-<kebnekaise/rackham>.py | |
| pandas_matplotlib-linreg-pretty-batch.sh | U(R), H (K) | pandas_matplotlib-linreg-pretty-batch-<kebnekaise/rackham>.py | |
| simple-lightgbm.sh | U, H | simple-lightgbm.py | | 

## Extra/other 

| Name | System | Related Python script | Comments |
| ---- | ------ | --------------------- | -------- |
| run_horovod.sh | U, H | Transfer_Learning_NLP_Horovod.py | | 
