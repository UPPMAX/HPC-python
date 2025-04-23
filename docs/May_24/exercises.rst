Exercises and demos
============================

Examples
--------

Load and run
############

You need the data-file [scottish_hills.csv](https://raw.githubusercontent.com/UPPMAX/HPC-python/main/Exercises/examples/programs/scottish_hills.csv). Download here or find in the ``Exercises/examples/programs`` directory in the files you got from cloning the repo.

    Since the exercise opens a plot, you need to login with ThinLinc (or otherwise have an x11 server running on your system and login with ``ssh -X ...``).

The exercise is modified from an example found on https://ourcodingclub.github.io/tutorials/pandas-python-intro/.

.. warning::

   **Not relevant if using UPPMAX. Only if you are using HPC2N!**

   You need to also load Tkinter. Use this:

   .. code-block:: console

      ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3

   In addition, you need to add the following two lines to the top of your python script/run them first in Python:

   .. code-block:: python

      import matplotlib
      matplotlib.use('TkAgg')

.. exercise:: Python example with packages pandas and matplotlib

   We are using Python version ``3.11.x``. To access the packages ``pandas`` and ``matplotlib``, you may need to load other modules, depending on the site where you are working.

   .. tabs::

      .. tab:: UPPMAX

         Here you only need to load the ``python`` module, as the relevant packages are included (as long as you are not using GPUs, but that is talked about later in the course). Thus, you just do:

        .. code-block:: console

           ml python/3.11.8

      .. tab:: HPC2N

         On Kebnekaise you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). These versions will work well together:

         .. code-block:: console

            ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

   1. From inside Python/interactive (if you are on Kebnekaise, mind the warning above):

      Start python and run these lines:

      .. code-block:: python

         import pandas as pd

      .. code-block:: python

         import matplotlib.pyplot as plt

      .. code-block:: python

         dataframe = pd.read_csv("scottish_hills.csv")

      .. code-block:: python

         x = dataframe.Height

      .. code-block:: python

         y = dataframe.Latitude

      .. code-block:: python

         plt.scatter(x, y)

      .. code-block:: python

         plt.show()

      If you change the last line to ``plt.savefig("myplot.png")`` then you will instead get a file ``myplot.png`` containing the plot. This is what you would do if you were running a python script in a batch job.

   2. As a Python script (if you are on Kebnekaise, mind the warning above):

      Copy and save this script as a file (or just run the file ``pandas_matplotlib-<system>.py`` that is located in the ``<path-to>/Exercises/examples/programs`` directory you got from the repo or copied. Where <system> is either ``rackham`` or ``kebnekaise``.

      .. tabs::

         .. tab:: rackham

            .. code-block:: python

               import pandas as pd
               import matplotlib.pyplot as plt

               dataframe = pd.read_csv("scottish_hills.csv")
               x = dataframe.Height
               y = dataframe.Latitude
               plt.scatter(x, y)
               plt.show()

         .. tab:: kebnekaise

            .. code-block:: python

               import pandas as pd
               import matplotlib
               import matplotlib.pyplot as plt

               matplotlib.use('TkAgg')

               dataframe = pd.read_csv("scottish_hills.csv")
               x = dataframe.Height
               y = dataframe.Latitude
               plt.scatter(x, y)
               plt.show()

Install packages
################

This is for the course environment and needed for one of the exercisesin the ML section.

Create a virtual environment called ``vpyenv``. First load the python version you want to base your virtual environment on, as well as the site-installed ML packages.

.. tabs::

   .. tab:: UPPMAX

      .. code-block:: console

          $ module load uppmax
          $ module load python/3.11.8
          $ module load python_ML_packages/3.11.8-cpu
          $ python -m venv --system-site-packages /proj/hpc-python/<user-dir>/vpyenv

      Activate it.

      .. code-block:: console

         $ source /proj/hpc-python/<user-dir>/vpyenv/bin/activate

      Note that your prompt is changing to start with (vpyenv) to show that you are within an environment.

      Install your packages with ``pip`` (``--user`` not needed as you are in your virtual environment) and (optionally) giving the correct versions, like:

      .. code-block:: console

         (vpyenv) $ pip install --no-cache-dir --no-build-isolation scikit-build-core cmake lightgbm

      The reason for the other packages (``scikit-build-core`` and ``cmake``) being installed is that they are prerequisites for ``lightgbm``.

      Check what was installed

      .. code-block:: console

         (vpyenv) $ pip list

      Deactivate it.

      .. code-block:: console

         (vpyenv) $ deactivate

      Everytime you need the tools available in the virtual environment you activate it as above, after loading the python module.

      .. code-block:: console

         $ source /proj/hpc-python/<user-dir>/vpyenv/bin/activate

      More on virtual environment: https://docs.python.org/3/tutorial/venv.html

   .. tab:: HPC2N

      **First go to the directory you want your environment in.**

      Load modules for Python, SciPy-bundle, matplotlib, create the virtual environment, activate the environment, and install lightgbm and scikit-learn (since the versions available are not compatible with this Python) on Kebnekaise at HPC2N

      .. code-block:: console

         $ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2
         $ python -m venv --system-site-packages vpyenv
         $ source vpyenv/bin/activate
         (vpyenv) $ pip install --no-cache-dir --no-build-isolation lightgbm scikit-learn

      Deactivating a virtual environment.

      .. code-block:: console

         (vpyenv) $ deactivate

      Every time you need the tools available in the virtual environment you activate it as above (after first loading the modules for Python, Python packages, and prerequisites)

      .. code-block:: console

         $ source vpyenv/bin/activate


Interactive
###########

.. admonition:: Example, Kebnekaise, Requesting 4 cores for 30 minutes, then running Python
    :class: dropdown

        .. code-block:: sh

            b-an01 [~]$ salloc -n 4 --time=00:30:00 -A hpc2n2024-052
            salloc: Pending job allocation 20174806
            salloc: job 20174806 queued and waiting for resources
            salloc: job 20174806 has been allocated resources
            salloc: Granted job allocation 20174806
            salloc: Waiting for resource configuration
            salloc: Nodes b-cn0241 are ready for job
            b-an01 [~]$ module load GCC/12.3.0 Python/3.11.3
            b-an01 [~]$

.. admonition:: Adding two numbers from user input (add2.py)
    :class: dropdown

        .. code-block:: python

            # This program will add two numbers that are provided by the user

            # Get the numbers
            a = int(input("Enter the first number: "))
            b = int(input("Enter the second number: "))

            # Add the two numbers together
            sum = a + b

            # Output the sum
            print("The sum of {0} and {1} is {2}".format(a, b, sum))

.. admonition:: Adding two numbers given as arguments (sum-2args.py)
    :class: dropdown

        .. code-block:: python

            import sys

            x = int(sys.argv[1])
            y = int(sys.argv[2])

            sum = x + y

            print("The sum of the two numbers is: {0}".format(sum))

Now for the examples:

.. admonition:: Example, Kebnekaise, Running a Python script in the allocation we made further up. Notice that since we asked for 4 cores, the script is run 4 times, since it is a serial script
    :class: dropdown

        .. code-block:: sh

            b-an01 [~]$ srun python sum-2args.py 3 4
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            b-an01 [~]$

.. admonition:: Example, Running a Python script in the above allocation, but this time a script that expects input from you.
    :class: dropdown

        .. code-block:: sh

            b-an01 [~]$ srun python add2.py
            2
            3
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5

Batch mode
##########

**Serial code**

This first example shows how to run a short, serial script. The batch script (named ``run_mmmult.sh``) can be found in the directory /HPC-Python/Exercises/examples/<center>, where <center> is hpc2n or uppmax. The Python script is in /HPC-Python/Exercises/examples/programs and is named ``mmmult.py``.

1. The batch script is run with ``sbatch run_mmmult.sh``.
2. Try type ``squeue -u <username>`` to see if it is pending or running.
3. When it has run, look at the output with ``nano slurm-<jobid>.out``.

.. tabs::

   .. tab:: UPPMAX

        Short serial example script for Rackham. Loading Python 3.11.8. Numpy is preinstalled and does not need to be loaded.

        .. code-block:: sh

            #!/bin/bash -l
            #SBATCH -A naiss2024-22-415 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here Python 3.11.8.
            module load python/3.11.8

            # Run your Python script
            python mmmult.py


   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2023.07 and Python/3.11.3

        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A hpc2n2024-052 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for Python/3.11.3 and compatible SciPy-bundle
            module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07

            # Run your Python script
            python mmmult.py


   .. tab:: mmmult.py

        Python example code

        .. code-block:: python

            import timeit
            import numpy as np

            starttime = timeit.default_timer()

            np.random.seed(1701)

            A = np.random.randint(-1000, 1000, size=(8,4))
            B = np.random.randint(-1000, 1000, size =(4,4))

            print("This is matrix A:\n", A)
            print("The shape of matrix A is ", A.shape)
            print()
            print("This is matrix B:\n", B)
            print("The shape of matrix B is ", B.shape)
            print()
            print("Doing matrix-matrix multiplication...")
            print()

            C = np.matmul(A, B)

            print("The product of matrices A and B is:\n", C)
            print("The shape of the resulting matrix is ", C.shape)
            print()
            print("Time elapsed for generating matrices and multiplying them is ", timeit.default_timer() - starttime)


**GPU code**

.. tabs::

   .. tab:: UPPMAX

        Short GPU example for running ``compute.py`` on Snowy.

        .. code-block:: sh

            #!/bin/bash -l
            #SBATCH -A naiss2024-22-415
            #SBATCH -t 00:10:00
            #SBATCH --exclusive
            #SBATCH -n 1
            #SBATCH -M snowy
            #SBATCH --gres=gpu=1

            # Load any modules you need, here loading python 3.11.8 and the ML packages
            module load uppmax
            module load python/3.11.8
            module load python_ML_packages/3.11.8-gpu

            # Run your code
            python compute.py


   .. tab:: HPC2N

        Example with running ``compute.py`` on Kebnekaise.

        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A hpc2n2024-052 # Change to your own
            #SBATCH --time=00:10:00  # Asking for 10 minutes
            # Asking for one V100 card
            #SBATCH --gres=gpu:v100:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 SciPy-bundle/2023.07 numba/0.58.1

            # Run your Python script
            python compute.py


   .. tab:: compute.py

        This Python script can (just like the batch scripts for UPPMAX and HPC2N), be found in the ``/HPC-Python/Exercises/examples`` directory, under the subdirectory ``programs`` - if you have cloned the repo or copied the tarball with the exercises.

        .. code-block:: python

           from numba import jit, cuda
           import numpy as np
           # to measure exec time
           from timeit import default_timer as timer

           # normal function to run on cpu
           def func a):
               for i in range(10000000):
                   a[i]+= 1

           # function optimized to run on gpu
           @jit(target_backend='cuda')
           def func2(a):
               for i in range(10000000):
                   a[i]+= 1
           if __name__=="__main__":
               n = 10000000
               a = np.ones(n, dtype = np.float64)

               start = timer()
               func(a)
               print("without GPU:", timer()-start)

               start = timer()
               func2(a)
               print("with GPU:", timer()-start)


.. challenge:: Run the first serial example script from further up on the page for this short Python code (sum-2args.py)

    .. code-block:: python

        import sys

        x = int(sys.argv[1])
        y = int(sys.argv[2])

        sum = x + y

        print("The sum of the two numbers is: {0}".format(sum))

    Remember to give the two arguments to the program in the batch script.

.. solution:: Solution for HPC2N
    :class: dropdown

          This batch script is for Kebnekaise. Adding the numbers 2 and 3.

          .. code-block:: sh

            #!/bin/bash
            #SBATCH -A hpc2n2024-052 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for Python 3.11.3
            module load GCC/12.3.0  Python/3.11.3

            # Run your Python script
            python sum-2args.py 2 3

.. solution:: Solution for UPPMAX
    :class: dropdown

          This batch script is for UPPMAX. Adding the numbers 2 and 3.

          .. code-block:: sh

            #!/bin/bash -l
            #SBATCH -A naiss2024-22-415 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for python 3.11.8
            module load python/3.11.8

            # Run your Python script
            python sum-2args.py 2 3


Machine Learning
################

**Pandas and matplotlib**

This is the same example that was shown in the section about loading and running Python, but now changed slightly to run as a batch job. The main difference is that here we cannot open the plot directly, but have to save to a file instead. You can see the change inside the Python script.

.. tabs::

   .. tab:: Directly

      Remove the # if running on Kebnekaise

      .. code-block::

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt

         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.show()

   .. tab:: From a Batch-job

      Remove the # if running on Kebnekaise. The script below can be found as ``pandas_matplotlib-batch.py`` or ``pandas_matplotlib-batch-kebnekaise.py`` in the ``Exercises/examples/programs`` directory.

      .. code-block::

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt

         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.savefig("myplot.png")

Batch scripts for running on Rackham and Kebnekaise.

.. tabs::

   .. tab:: Rackham

      .. code-block::

         #!/bin/bash -l
         #SBATCH -A naiss2024-22-415
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.8
         ml python/3.11.8

         # Run your Python script
         python pandas_matplotlib-batch.py

   .. tab:: Kebnekaise

      .. code-block::

         #!/bin/bash
         #SBATCH -A hpc2n2024-052
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.3
         ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

         # Run your Python script
         python pandas_matplotlib-batch-kebnekaise.py

Submit with ``sbatch <batch-script.sh>``.

The batch scripts can be found in the directories for hpc2n and uppmax, under ``Exercises/examples/``, and they are named ``pandas_matplotlib-batch.sh`` and ``pandas_matplotlib-batch-kebnekaise.sh``.


**PyTorch**

.. admonition:: We use PyTorch Tensors to fit a third order polynomial to a sine function. The forward and backward passes through the network are manually implemented.
    :class: dropdown

        .. code-block:: python

            # -*- coding: utf-8 -*-

            import torch
            import math

            dtype = torch.float
            device = torch.device("cpu")
            device = torch.device("cuda:0") # Comment this out to not run on GPU

            # Create random input and output data
            x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
            y = torch.sin(x)

            # Randomly initialize weights
            a = torch.randn((), device=device, dtype=dtype)
            b = torch.randn((), device=device, dtype=dtype)
            c = torch.randn((), device=device, dtype=dtype)
            d = torch.randn((), device=device, dtype=dtype)

            learning_rate = 1e-6
            for t in range(2000):
                # Forward pass: compute predicted y
                y_pred = a + b * x + c * x ** 2 + d * x ** 3

                # Compute and print loss
                loss = (y_pred - y).pow(2).sum().item()
                if t % 100 == 99:
                    print(t, loss)

                # Backprop to compute gradients of a, b, c, d with respect to loss
                grad_y_pred = 2.0 * (y_pred - y)
                grad_a = grad_y_pred.sum()
                grad_b = (grad_y_pred * x).sum()
                grad_c = (grad_y_pred * x ** 2).sum()
                grad_d = (grad_y_pred * x ** 3).sum()

                # Update weights using gradient descent
                a -= learning_rate * grad_a
                b -= learning_rate * grad_b
                c -= learning_rate * grad_c
                d -= learning_rate * grad_d

            print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


In order to run this at HPC2N/UPPMAX you should either do a batch job or run interactively on compute nodes. Remember, you should not run long/resource heavy jobs on the login nodes, and they also do not have GPUs if you want to use that.

This is an example of a batch script for running the above example, using PyTorch 2.1.x and Python 3.11.x, and running on GPUs.

.. admonition:: Example batch script, running on Kebnekaise
    :class: dropdown

        .. code-block:: sh

            #!/bin/bash
            # Remember to change this to your own project ID after the course!
            #SBATCH -A hpc2n2024-052
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # The following two lines splits the output in a file for any errors and a file for other output.
            #SBATCH --error=job.%J.err
            #SBATCH --output=job.%J.out
            # Asking for one V100
            #SBATCH --gres=gpu:V100:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

            srun python pytorch_fitting_gpu.py


.. admonition:: UPPMAX as run in an interactive Snowy session
    :class: dropdown

        .. code-block:: sh

            $ interactive -A naiss2024-22-415 -n 1 -M snowy --gres=gpu:1  -t 1:00:01
            You receive the high interactive priority.

            Please, use no more than 8 GB of RAM.

            Waiting for job 6907137 to start...
            Starting job now -- you waited for 90 seconds.

            $  ml uppmax
            $  ml python/3.11.8
            $  module load python_ML_packages/3.11.8-gpu
            $  cd /proj/naiss2024-22-415/<user-dir>/HPC-python/Exercises/examples/programs
            $ srun python pytorch_fitting_gpu.py
            99 134.71942138671875
            199 97.72868347167969
            299 71.6167221069336
            399 53.178802490234375
            499 40.15779113769531
            599 30.9610652923584
            699 24.464630126953125
            799 19.875120162963867
            899 16.632421493530273
            999 14.341087341308594
            1099 12.721846580505371
            1199 11.577451705932617
            1299 10.76859188079834
            1399 10.196844100952148
            1499 9.792669296264648
            1599 9.506935119628906
            1699 9.304922103881836
            1799 9.162087440490723
            1899 9.061092376708984
            1999 8.989676475524902
            Result: y = 0.013841948471963406 + 0.855550229549408 x + -0.002387965563684702 x^2 + -0.09316103905439377 x^3



TensorFlow
----------

The example comes from https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/ but there are also good examples at https://www.tensorflow.org/tutorials

We are using Tensorflow 2.11.0-CUDA-11.7.0 (and Python 3.10.4) at HPC2N, since that is the newest GPU-enabled TensorFlow currently installed there.

On UPPMAX we are using TensorFlow 2.15.0 (included in python_ML_packages/3.11.8-gpu) and Python 3.11.8.

.. tabs::

   .. tab:: HPC2N

      Since we need scikit-learn, we are also loading the scikit-learn/1.1.2 which is compatible with the other modules we are using.

      Thus, load modules: ``GCC/11.3.0  OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2`` in your batch script.

   .. tab:: UPPMAX

      UPPMAX has scikit-learn in the python_ML_packages, so we do not need to load anything extra there.

        - Load modules: ``module load uppmax python/3.11.8 python_ML_packages/3.11.8-gpu``
           - On Rackham we should use python_ML-packages/3.11.8-cpu, while on a GPU node the GPU version should be loaded (like we do in this example, which will work either in a batch script submitted to Snowy or in an interactive job running on Snowy).

.. admonition:: We will work with this example
    :class: dropdown

        .. code-block:: sh

            # mlp for binary classification
            from pandas import read_csv
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense
            # load the dataset
            path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
            df = read_csv(path, header=None)
            # split into input and output columns
            X, y = df.values[:, :-1], df.values[:, -1]
            # ensure all data are floating point values
            X = X.astype('float32')
            # encode strings to integer
            y = LabelEncoder().fit_transform(y)
            # split into train and test datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            # determine the number of input features
            n_features = X_train.shape[1]
            # define model
            model = Sequential()
            model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
            model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
            model.add(Dense(1, activation='sigmoid'))
            # compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # fit the model
            model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
            # evaluate the model
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print('Test Accuracy: %.3f' % acc)
            # make a prediction
            row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
            yhat = model.predict([row])
            print('Predicted: %.3f' % yhat)


In order to run the above example, we will create a batch script and submit it.

.. admonition:: Example batch script for Kebnekaise, TensorFlow version 2.11.0 and Python version 3.10.4, and scikit-learn 1.1.2
    :class: dropdown

        .. code-block:: sh

            #!/bin/bash
            # Remember to change this to your own project ID after the course!
            #SBATCH -A hpc2n2024-052
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one V100 GPU
            #SBATCH --gres=gpu:v100:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/11.3.0 Python/3.10.4 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2

            # Run your Python script
            python example-tf.py


.. admonition:: Example batch script for Snowy, Python version 3.11.8, and the python_ML_packages/3.11.8-gpu containing Tensorflow
    :class: dropdown

      .. code-block:: sh

            #!/bin/bash -l
            # Remember to change this to your own project ID after the course!
            #SBATCH -A naiss2024-22-415
            # We want to run on Snowy
            #SBATCH -M snowy
            # We are asking for 15 minutes
            #SBATCH --time=00:15:00
            #SBATCH --gres=gpu:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load uppmax
            module load python_ML_packages/3.11.8-gpu

            # Run your Python script
            python example-tf.py




Submit with ``sbatch <myjobscript.sh>``. After submitting you will (as usual) be given the job-id for your job. You can check on the progress of your job with ``squeue -u <username>`` or ``scontrol show <job-id>``.

Note: if you are logged in to Rackham on UPPMAX and have submitted a GPU job to Snowy, then you need to use this to see the job queue:

``squeue -M snowy -u <username>``

The output and errors will in this case be written to ``slurm-<job-id>.out``.


General
#######

You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes.

Running several jobs from within one job
''''''''''''''''''''''''''''''''''''''''

This example shows how you would run several programs or variations of programs sequentially within the same job:

.. tabs::

   .. tab:: HPC2N

      Example batch script for Kebnekaise, TensorFlow version 2.11.0 and Python version 3.11.3

      .. code-block:: sh

         #!/bin/bash
         # Remember to change this to your own project ID after the course!
         #SBATCH -A hpc2n2024-052
         # We are asking for 5 minutes
         #SBATCH --time=00:05:00
         # Asking for one V100
         #SBATCH --gres=gpu:v100:1
         # Remove any loaded modules and load the ones we need
         module purge  > /dev/null 2>&1
         module load GCC/10.3.0 OpenMPI/4.1.1 SciPy-bundle/2021.05 TensorFlow/2.6.0-CUDA-11.3-1
         # Output to file - not needed if your job creates output in a file directly
         # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
         python <my_tf_program.py> <param1> <param2> > myoutput1 2>&1
         cp myoutput1 mydatadir
         python <my_tf_program.py> <param3> <param4> > myoutput2 2>&1
         cp myoutput2 mydatadir
         python <my_tf_program.py> <param5> <param6> > myoutput3 2>&1
         cp myoutput3 mydatadir

   .. tab:: UPPMAX

      Example batch script for Snowy, TensorFlow version 2.15 and Python version 3.11.8.

      .. code-block:: sh

         #!/bin/bash -l
         # Remember to change this to your own project ID after the course!
         #SBATCH -A naiss2024-22-415
         # We are asking for at least 1 hour
         #SBATCH --time=01:00:01
         #SBATCH -M snowy
         #SBATCH --gres=gpu:1

         # Remove any loaded modules and load the ones we need
         module purge  > /dev/null 2>&1
         module load uppmax
         module load python_ML_packages/3.11.8-gpu
         # Output to file - not needed if your job creates output in a file directly
         # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
         python tf_program.py 1 2 > myoutput1 2>&1
         cp myoutput1 mydatadir
         python tf_program.py 3 4 > myoutput2 2>&1
         cp myoutput2 mydatadir
         python tf_program.py 5 6 > myoutput3 2>&1
         cp myoutput3 mydatadir


The challenge here is to adapt the above batch scripts to suitable python scripts and directories.

.. challenge::

   Try to modify the files ``pandas_matplotlib-linreg-<rackham/kebnekaise>.py`` and ``pandas_matplotlib-linreg-pretty-<rackham/kebnekaise>.py so they could be run from a batch job (change the pop-up plots to save-to-file).

   Also change the batch script ``pandas_matplotlib.sh`` (or ``pandas_matplotlib-kebnekaise.sh``) to run your modified python codes.

.. challenge::

   In this exercise you will be using the course environment that you prepared in the "Install packages" section (here: https://uppmax.github.io/HPC-python/install_packages.html#prepare-the-course-environment).

   You will run the Python code ``simple_lightgbm.py`` found in the ``Exercises/examples/programs`` directory. The code was taken from https://github.com/microsoft/LightGBM/tree/master and lightly modified.

   Try to write a batch script that runs this code. Remember to activate the course environment.

   .. tabs::

      .. tab:: simple_lightgbm.py

         .. code-block::

            # coding: utf-8
            from pathlib import Path

            import pandas as pd
            from sklearn.metrics import mean_squared_error

            import lightgbm as lgb

            print("Loading data...")
            # load or create your dataset
            df_train = pd.read_csv(str("regression.train"), header=None, sep="\t")
            df_test = pd.read_csv(str("regression.test"), header=None, sep="\t")

            y_train = df_train[0]
            y_test = df_test[0]
            X_train = df_train.drop(0, axis=1)
            X_test = df_test.drop(0, axis=1)

            # create dataset for lightgbm
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # specify your configurations as a dict
            params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": {"l2", "l1"},
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": 0,
            }

            print("Starting training...")
            # train
            gbm = lgb.train(
                params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)]
            )

            print("Saving model...")
            # save model to file
            gbm.save_model("model.txt")

            print("Starting predicting...")
            # predict
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            # eval
            rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
            print(f"The RMSE of prediction is: {rmse_test}")

      .. tab:: Rackham

         .. admonition:: Click to reveal the solution!
             :class: dropdown

                   .. code-block::

                      #!/bin/bash -l
                      # Change to your own project ID after the course!
                      #SBATCH -A naiss2024-22-415
                      # We are asking for 10 minutes
                      #SBATCH --time=00:10:00
                      #SBATCH -n 1

                      # Set a path where the example programs are installed.
                      # Change the below to your own path to where you placed the example programs
                      MYPATH=/proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/
                      # Activate the course environment (assuming it was called vpyenv)
                      source /proj/hpc-python/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate
                      # Remove any loaded modules and load the ones we need
                      module purge  > /dev/null 2>&1
                      module load uppmax
                      module load python/3.11.8

                      # Run your Python script
                      python $MYPATH/simple_lightgbm.py

      .. tab:: Kebnekaise

         .. admonition:: Click to reveal the solution!
             :class: dropdown

                   .. code-block::

                      #!/bin/bash
                      # Change to your own project ID after the course!
                      #SBATCH -A hpc2n2024-052
                      # We are asking for 10 minutes
                      #SBATCH --time=00:10:00
                      #SBATCH -n 1

                      # Set a path where the example programs are installed.
                      # Change the below to your own path to where you placed the example programs
                      MYPATH=/proj/nobackup/python-hpc/<mydir-name>/HPC-python/Exercises/examples/programs/

                      # Remove any loaded modules and load the ones we need
                      module purge  > /dev/null 2>&1
                      module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

                      # Activate the course environment (assuming it was called vpyenv)
                      source /proj/nobackup/python-hpc/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate

                      # Run your Python script
                      python $MYPATH/simple_lightgbm.py

GPU
###

Numba is installed as a module at HPC2N, but not in a version compatible with the Python we are using in this course (3.10.4), so we will have to install it ourselves. The process is the same as in the examples given for the isolated/virtual environment, and we will be using the virtual environment created earlier here. We also need numpy, so we are loading SciPy-bundle as we have done before:

.. admonition:: Load Python 3.10.4 and its prerequisites + SciPy-bundle + CUDA, then activate the virtual environment before installing numba
    :class: dropdown

        .. code-block:: sh

             $ module load GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 SciPy-bundle/2021.10 CUDA/11.7.0
             $ python -m venv --system-site-packages vpyenv
             $ source /proj/nobackup/python-hpc/bbrydsoe/vpyenv/bin/activate
             (vpyenv) $ pip install --no-cache-dir --no-build-isolation numba
             Collecting numba
               Downloading numba-0.56.0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.5 MB)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.5/3.5 MB 38.7 MB/s eta 0:00:00
             Requirement already satisfied: setuptools in /pfs/proj/nobackup/fs/projnb10/python-hpc/bbrydsoe/vpyenv/lib/python3.9/site-packages (from numba) (63.1.0)
             Requirement already satisfied: numpy<1.23,>=1.18 in /cvmfs/ebsw.hpc2n.umu.se/amd64_ubuntu2004_bdw/software/SciPy-bundle/2021.05-foss-2021a/lib/python3.9/site-packages (from numba) (1.20.3)
             Collecting llvmlite<0.40,>=0.39.0dev0
               Downloading llvmlite-0.39.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.6 MB)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 34.6/34.6 MB 230.0 MB/s eta 0:00:00
             Installing collected packages: llvmlite, numba
             Successfully installed llvmlite-0.39.0 numba-0.56.0

             [notice] A new release of pip available: 22.1.2 -> 22.2.2
             [notice] To update, run: pip install --upgrade pip


        Let us try using it. We are going to use the following program for testing (it was taken from https://linuxhint.com/gpu-programming-python/ but there are also many great examples at https://numba.readthedocs.io/en/stable/cuda/examples.html):

.. admonition:: Python example using Numba
    :class: dropdown

        .. code-block:: python

             import numpy as np
             from timeit import default_timer as timer
             from numba import vectorize

             # This should be a substantially high value.
             NUM_ELEMENTS = 100000000

             # This is the CPU version.
             def vector_add_cpu(a, b):
               c = np.zeros(NUM_ELEMENTS, dtype=np.float32)
               for i in range(NUM_ELEMENTS):
                   c[i] = a[i] + b[i]
               return c

             # This is the GPU version. Note the @vectorize decorator. This tells
             # numba to turn this into a GPU vectorized function.
             @vectorize(["float32(float32, float32)"], target='cuda')
             def vector_add_gpu(a, b):
               return a + b;

             def main():
               a_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
               b_source = np.ones(NUM_ELEMENTS, dtype=np.float32)

               # Time the CPU function
               start = timer()
               vector_add_cpu(a_source, b_source)
               vector_add_cpu_time = timer() - start

               # Time the GPU function
               start = timer()
               vector_add_gpu(a_source, b_source)
               vector_add_gpu_time = timer() - start

                # Report times
                print("CPU function took %f seconds." % vector_add_cpu_time)
                print("GPU function took %f seconds." % vector_add_gpu_time)

                return 0

             if __name__ == "__main__":
               main()

As before, we need a batch script to run the code. There are no GPUs on the login node.

.. admonition:: Batch script to run the numba code (add-list.py) at Kebnekaise
    :class: dropdown

        .. code-block:: sh

            #!/bin/bash
            # Remember to change this to your own project ID after the course!
            #SBATCH -A hpc2nXXXX-YYY
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one K80
            #SBATCH --gres=gpu:k80:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 SciPy-bundle/2021.10 CUDA/11.7.0

            # Activate the virtual environment we installed to
            source /proj/nobackup/support-hpc2n/bbrydsoe/vpyenv/bin/activate

            # Run your Python script
            python add-list.py


As before, submit with ``sbatch add-list.sh`` (assuming you called the batch script thus - change to fit your own naming style).

Numba example 2
---------------

An initial implementation of the 2D integration problem with the CUDA support for Numba could be
as follows:

   .. admonition:: ``integration2d_gpu.py``
      :class: dropdown

      .. code-block:: python

         from __future__ import division
         from numba import cuda, float32
         import numpy
         import math
         from time import perf_counter

         # grid size
         n = 100*1024
         threadsPerBlock = 16
         blocksPerGrid = int((n+threadsPerBlock-1)/threadsPerBlock)

         # interval size (same for X and Y)
         h = math.pi / float(n)

         @cuda.jit
         def dotprod(C):
             tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

             if tid >= n:
                 return

             #cummulative variable
             mysum = 0.0
             # fine-grain integration in the X axis
             x = h * (tid + 0.5)
             # regular integration in the Y axis
             for j in range(n):
                 y = h * (j + 0.5)
                 mysum += math.sin(x + y)

             C[tid] = mysum


         # array for collecting partial sums on the device
         C_global_mem = cuda.device_array((n),dtype=numpy.float32)

         starttime = perf_counter()
         dotprod[blocksPerGrid,threadsPerBlock](C_global_mem)
         res = C_global_mem.copy_to_host()
         integral = h**2 * sum(res)
         endtime = perf_counter()

         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

The time for executing the kernel and doing some postprocessing to the outputs (copying
the C array and doing a reduction)  was 4.35 sec. which is a much smaller value than the
time for the serial numba code of 152 sec.

Notice the larger size of the grid in the present case (100*1024) compared to the
serial case's size we used previously (10000). Large computations are necessary on the GPUs
to get the benefits of this architecture.

One can take advantage of the shared memory in a thread block to write faster code. Here,
we wrote the 2D integration example from the previous section where threads in a block
write on a `shared[]` array. Then, this array is reduced (values added) and the output is
collected in the array ``C``. The entire code is here:


   .. admonition:: ``integration2d_gpu_shared.py``
      :class: dropdown

      .. code-block:: python

         from __future__ import division
         from numba import cuda, float32
         import numpy
         import math
         from time import perf_counter

         # grid size
         n = 100*1024
         threadsPerBlock = 16
         blocksPerGrid = int((n+threadsPerBlock-1)/threadsPerBlock)

         # interval size (same for X and Y)
         h = math.pi / float(n)

         @cuda.jit
         def dotprod(C):
             # using the shared memory in the thread block
             shared = cuda.shared.array(shape=(threadsPerBlock), dtype=float32)

             tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
             shrIndx = cuda.threadIdx.x

             if tid >= n:
                 return

             #cummulative variable
             mysum = 0.0
             # fine-grain integration in the X axis
             x = h * (tid + 0.5)
             # regular integration in the Y axis
             for j in range(n):
                 y = h * (j + 0.5)
                 mysum += math.sin(x + y)

             shared[shrIndx] = mysum

             cuda.syncthreads()

             # reduction for the whole thread block
             s = 1
             while s < cuda.blockDim.x:
                 if shrIndx % (2*s) == 0:
                     shared[shrIndx] += shared[shrIndx + s]
                 s *= 2
                 cuda.syncthreads()
             # collecting the reduced value in the C array
             if shrIndx == 0:
                 C[cuda.blockIdx.x] = shared[0]

         # array for collecting partial sums on the device
         C_global_mem = cuda.device_array((blocksPerGrid),dtype=numpy.float32)

         starttime = perf_counter()
         dotprod[blocksPerGrid,threadsPerBlock](C_global_mem)
         res = C_global_mem.copy_to_host()
         integral = h**2 * sum(res)
         endtime = perf_counter()

         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

We need a batch script to run this Python code, an example script is here:


.. code-block:: sh

    #!/bin/bash
    #SBATCH -A project_ID
    #SBATCH -t 00:05:00
    #SBATCH -N 1
    #SBATCH -n 28
    #SBATCH -o output_%j.out   # output file
    #SBATCH -e error_%j.err    # error messages
    #SBATCH --gres=gpu:k80:2
    #SBATCH --exclusive

    ml purge > /dev/null 2>&1
    ml GCCcore/11.2.0 Python/3.9.6
    ml GCC/11.2.0 OpenMPI/4.1.1
    ml CUDA/11.7.0

    virtualenv --system-site-packages /proj/nobackup/<your-project-storage>/vpyenv-python-course
    source /proj/nobackup/<your-project-storage>/vpyenv-python-course/bin/activate

    python integration2d_gpu.py

The simulation time for this problem's size
was 1.87 sec.


Exercises
---------

.. challenge:: Run the first serial example from further up on the page for this short Python code (sum-2args.py)

    .. code-block:: python

        import sys

        x = int(sys.argv[1])
        y = int(sys.argv[2])

        sum = x + y

        print("The sum of the two numbers is: {0}".format(sum))

    Remember to give the two arguments to the program in the batch script.

.. solution::
    :class: dropdown

          This is for Kebnekaise. Adding the numbers 2 and 3.

          .. code-block:: sh

            #!/bin/bash
            #SBATCH -A hpc2nXXXX-YYY # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for Python 3.9.6
            module load GCC/11.2.0  OpenMPI/4.1.1 Python/3.9.6

            # Run your Python script
            python sum-2args.py 2 3
