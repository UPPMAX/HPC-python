Using Python for Machine Learning jobs
======================================

While Python does not run fast, it is still well suited for machine learning. However, it is fairly easy to code in, and this is particularly useful in machine learning where the right solution is rarely known from the start. A lot of tests and experimentation is needed, and the program usually goes through many iterations. In addition, there are a lot of useful libraries written for machine learning in Python, making it a good choice for this area. 

Some of the most used libraries in Python for machine learning are: 

- PyTorch
- scikit-learn
- TensorFlow

These are all available at UPPMAX and HPC2N. 

In this course we will look at two examples: PyTorch and TensorFlow, and show how you run them at our centres. 

PyTorch
-------

PyTorch has: 

- An n-dimensional Tensor, similar to numpy, but can run on GPUs
- Automatic differentiation for building and training neural networks

The example we will use in this course is taken from the official PyTorch page: https://pytorch.org/ and the problem is of fitting :math:`y=sin⁡(x)` with a third order polynomial. 

.. admonition:: We use PyTorch Tensors to fit a third order polynomial to a sine function. The forward and backward passes through the network are manually implemented. 
    :class: dropdown

        .. code-block:: python
        
            # -*- coding: utf-8 -*-
            
            import torch
            import math
            
            dtype = torch.float
            device = torch.device("cpu")
            # device = torch.device("cuda:0") # Uncomment this to run on GPU
            
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

You can find the full list of examples for this problem here: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

In order to run this at HPC2N (and at UPPMAX?) you should use a batch job. 

This is an example of a batch script for running the above example, using PyTorch 1.10.0 and Python 3.9.5, running on GPUs. 

.. admonition:: Example batch script, running the above example (assuming it is named pytorch_fitting_gpu.py) 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A SNIC2022-22-641
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # The following two lines splits the output in a file for any errors and a file for other output. 
            #SBATCH --error=job.%J.err
            #SBATCH --output=job.%J.out
            # Asking for one K80
            #SBATCH --gres=gpu:k80:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 PyTorch/1.10.0-CUDA-11.3.1
            
            srun python pytorch_fitting_gpu.py



TensorFlow
----------
