Using GPUs with Python
======================

Not every Python program is suitable for GPU acceleration. GPUs processes simple functions very fast, and are best suited for repetitive and highly-parallel computing tasks. 

GPUs are originally designed to render high-resolution images and video concurrently and fast, but since they can perform parallel operations on multiple sets of data, they are also often used for other, non-graphical tasks. Common uses are machine learning and scientific computation were the GPUs can take advantage of massive parallelism. 

Many Python packages are not CUDA aware, but some have been written specifically with GPUs in mind. 

If you are usually working with for instance NumPy and SciPy, you could optimize your code for GPU computing by using CuPy which mimics most of the NumPy functions. Another option is using Numba, which has bindings to CUDA and lets you write CUDA kernels in Python yourself. This means you can use custom algorithms. 

One of the most common use of GPUs with Python is for machine learning or deep learning. For these cases you would use something like Tensorflow or PyTorch - libraries which can handle CPU and GPU processing internally without the programmer needing to do so. 

Numba example
-------------

Numba is installed as a module at HPC2N, but not in a version compatible with the Python we are using in this course (3.9.5), so we will have to install it ourselves. The process is the same as in the examples given for the isolated/virtual environment, and we will be using the virtual environment created earlier here: 

