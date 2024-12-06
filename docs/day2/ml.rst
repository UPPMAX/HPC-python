Machine Learning and Deep Learning
========================================================

.. questions::

   - Which machine learning and deep learning tools are installed at HPC2N, UPPMAX, and LUNARC?
   - How to start the tools at HPC2N, UPPMAX, and LUNARC?
   - How to deploy GPU:s with ML/DL at HPC2N, UPPMAX, and LUNARC?

.. objectives::

   - Get a general overview of ML/DL with Python. 
   - Get a general overview of installed ML/DL tools at HPC2N, UPPMAX, and LUNARC.
   - Get started with ML/DL in Python.
   - Code along and demos (Kebnekaise, Rackham/Snowy, Cosmos and Tetralith).
   - We will not learn about:
      - How to write and optimize ML/DL code.
      - How to use multi-node setup for training models on CPU and GPU.  


Introduction
------------------
   
   Python is well suited for machine learning and deep learning. For instance, it is fairly easy to code in, and this is particularly useful in ML/DL where the right solution is rarely known from the start. A lot of tests and experimentation is needed, and the program usually goes through many iterations. In addition, there are a lot of useful libraries written for ML and DL in Python, making it a good choice for this area.  

   Some of the most used libraries in Python for ML/DL are: 

   - scikit-learn (sklearn)
   - PyTorch
   - TensorFlow

Comparison of ML/DL Libraries
-----------------------------

.. list-table:: 
   :widths: 20 20 20 20
   :header-rows: 1

   * - Feature
     - scikit-learn
     - PyTorch
     - TensorFlow
   * - Primary Use
     - Traditional machine learning
     - Deep learning and neural networks
     - Deep learning and neural networks
   * - Ease of Use
     - High, simple API
     - Moderate, more control over computations
     - Moderate, high-level Keras API available
   * - Performance
     - Good for small to medium datasets
     - Excellent with GPU support
     - Excellent with GPU support
   * - Flexibility
     - Limited to traditional ML algorithms
     - High, supports dynamic computation graphs
     - High, supports both static and dynamic computation graphs
   * - Community and Support
     - Large, extensive documentation
     - Large, growing rapidly
     - Large, extensive documentation and community support

These are all available at UPPMAX, HPC2N, and LUNARC. 

In this course we will look at examples for these, and show how you run them at our centres. 

.. admonition:: Learning Material
   :class: dropdown

   For more beginner friendly examples and detailed tutorials, you can visit the following resources:

   - PyTorch examples: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
   - TensorFlow tutorials: https://www.tensorflow.org/tutorials
   - Scikit-Learn basics: https://scikit-learn.org/stable/getting_started.html
   - Machine Learning with Python: https://machinelearningmastery.com/start-here/#python

   For more advanced users, you can visit the following resources:

   - Pytorch data parallelism: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
   - TensorFlow distributed_training: https://www.tensorflow.org/guide/keras/distributed_training
   - Scikit-Learn parallelism: https://scikit-learn.org/stable/computing/parallelism.html#parallelism
   - Ray cluster for hyperparameter tuning: https://docs.ray.io/en/latest/ray-more-libs/joblib.html

   
List of installed ML/DL tools
############################# 

There are minor differences depending on the version of python. 

The list is not exhaustive, but lists the more popular ML/DL libraries. I encourage you to `module spider` them to see the exact versions before loading them.

.. list-table::
   :widths: 15 30 30 15 10
   :header-rows: 1

   * - Tool
     - UPPMAX (python 3.11.8)
     - HPC2N (Python 3.11.3/3.11.5)
     - LUNARC (Python 3.11.3/3.11.5)
     - NSC (Python 3.11.3/3.11.5)
   * - NumPy
     - python
     - SciPy-bundle
     - SciPy-bundle
     - N.A.
   * - SciPy
     - python
     - SciPy-bundle
     - SciPy-bundle
     - N.A.
   * - Scikit-Learn (sklearn)
     - python_ML_packages (Python 3.11.8-gpu and Python 3.11.8-cpu) 
     - scikit-learn (no newer than for GCC/12.3.0 and Python 3.11.3)  
     - scikit-learn 
     - N.A.
   * - Theano
     - N.A.
     - Theano (only for some older Python versions)
     - N.A.
     - N.A. 
   * - TensorFlow
     - python_ML_packages (Python 3.11.8-gpu and Python 3.11.8-cpu)
     - TensorFlow (newest version is for Python 3.11.3)
     - TensorFlow (up to Python 3.10.4) 
     - N.A.
   * - Keras
     - python_ML_packages (Python 3.11.8-gpu and Python 3.11.8-cpu)
     - Keras (up to Python 3.8.6), TensorFlow (Python 3.11.3)
     - TensorFlow (up to Python 3.10.4)
     - N.A.
   * - PyTorch (torch)
     - python_ML_packages (Python 3.11.5-gpu and Python 3.11.8-cpu)
     - PyTorch (up to Python 3.11.3) 
     - PyTorch (up to Python 3.10.4) 
     - N.A.
   * - Pandas
     - python
     - SciPy-bundle
     - SciPy-bundle
     - N.A.
   * - Matplotlib
     - python
     - matplotlib
     - matplotlib
     - N.A.
   * - Beautiful Soup (beautifulsoup4)
     - python_ML_packages (Python 3.9.5-gpu and Python 3.11.8-cpu)
     - BeautifulSoup
     - BeautifulSoup
     - N.A.
   * - Seaborn
     - python_ML_packages (Python 3.9.5-gpu and Python 3.11.8-cpu)
     - Seaborn
     - Seaborn 
     - N.A.
   * - Horovod 
     - N.A.
     - Horovod (up to Python 3.11.3)
     - N.A.
     - N.A.    

Scikit-Learn
-------------

Scikit-learn (sklearn) is a powerful and easy-to-use open-source machine learning library for Python. It provides simple and efficient tools for data mining and data analysis, and it is built on NumPy, SciPy, and matplotlib. Scikit-learn is designed to interoperate with the Python numerical and scientific libraries.

More often that not, scikit-learn is used along with other popular libraries like tensorflow and pytorch to perform exploratory data analysis, data preprocessing, model selection, and evaluation. For our examples, we will use jupyter notebook on a CPU node to see visualization of the data and the results.

.. admonition:: Components of Scikit-learn
   :class: dropdown

   .. list-table::
      :widths: 20 40 40
      :header-rows: 1

      * - **Component**
        - **Definition**
        - **Examples**
      
      * - Estimators
        - Estimators are the core objects in scikit-learn. They implement algorithms for classification, regression, clustering, and more. An estimator is any object that learns from data; it implements the ``fit`` method, which is used to train the model.
        - 
         - ``LinearRegression`` for linear regression
         - ``KNeighborsClassifier`` for k-nearest neighbors classification
         - ``DecisionTreeClassifier`` for decision tree classification
      
      * - Transformers
        - Transformers are used for data preprocessing and feature extraction. They implement the ``fit`` and ``transform`` methods. The ``fit`` method learns the parameters from the data, and the ``transform`` method applies the transformation to the data.
        - 
            - ``StandardScaler`` for standardizing features by removing the mean and scaling to unit variance
            - ``PCA`` (Principal Component Analysis) for dimensionality reduction
            - ``TfidfVectorizer`` for converting a collection of raw documents to a matrix of TF-IDF features
      
      * - Pipelines
        - Pipelines are a way to streamline a machine learning workflow by chaining together multiple steps into a single object. A pipeline can include both transformers and estimators. This ensures that all steps are executed in the correct order and simplifies the process of parameter tuning.
        - A pipeline that standardizes the data and then applies a linear regression model:
         
            .. code-block:: python
            
               from sklearn.pipeline import Pipeline
               from sklearn.preprocessing import StandardScaler
               from sklearn.linear_model import LinearRegression

               pipeline = Pipeline([
                  ('scaler', StandardScaler()),
                  ('regressor', LinearRegression())
               ])
         
      * - Datasets
        - Scikit-learn provides several built-in datasets for testing and experimenting with machine learning algorithms. These datasets can be loaded using the `datasets` module.
        - 
            - ``load_iris`` for the Iris flower dataset
            - ``load_digits`` for the handwritten digits dataset
            - ``load_boston`` for the Boston house prices dataset

            Example of loading a dataset:
         
            .. code-block:: python
            
               from sklearn.datasets import load_iris

               iris = load_iris()
               X, y = iris.data, iris.target
         
      * - Model Evaluation
        - Scikit-learn provides various tools for evaluating the performance of machine learning models. These include metrics for classification, regression, and clustering, as well as methods for cross-validation.
        - 
            - ``accuracy_score`` for classification accuracy
            - ``mean_squared_error`` for regression error
            - ``silhouette_score`` for clustering quality
         
            Example of evaluating a model:
            
            .. code-block:: python
               
               from sklearn.metrics import accuracy_score
               from sklearn.model_selection import train_test_split
               from sklearn.neighbors import KNeighborsClassifier

               X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
               model = KNeighborsClassifier()
               model.fit(X_train, y_train)
               y_pred = model.predict(X_test)
               accuracy = accuracy_score(y_test, y_pred)
               print(f'Accuracy: {accuracy:.2f}')
            
      * - Parameter Searches
        - Scikit-learn provides tools for hyperparameter tuning, such as ``GridSearchCV`` and ``RandomizedSearchCV``. These tools help in finding the best parameters for a given model by performing an exhaustive search over specified parameter values.
        - Example of a parameter search:
         
            .. code-block:: python
               
               from sklearn.model_selection import GridSearchCV
               from sklearn.svm import SVC

               param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
               grid_search = GridSearchCV(SVC(), param_grid, cv=5)
               grid_search.fit(X_train, y_train)
               print(f'Best parameters: {grid_search.best_params_}')
               print(f'Best score: {grid_search.best_score_}')
         

Scikit-learn provides a comprehensive suite of tools for building and evaluating machine learning models, making it an essential library for data scientists and machine learning practitioners.

.. tabs::

   .. tab:: Example 1: Linear Regression

      .. code-block:: python

         import numpy as np
         import matplotlib.pyplot as plt
         from sklearn.linear_model import LinearRegression

         # Generate some data
         X = np.array([[1], [2], [3], [4], [5]])
         y = np.array([1, 3, 2, 3, 5])

         # Create and fit the model
         model = LinearRegression()
         model.fit(X, y)

         # Make predictions
         y_pred = model.predict(X)

         # Plot the results
         plt.scatter(X, y, color='black')
         plt.plot(X, y_pred, color='blue', linewidth=3)
         plt.xlabel('X')
         plt.ylabel('y')
         plt.title('Linear Regression Example')
         plt.show()

   .. tab:: Example 2: K-Nearest Neighbors

      .. code-block:: python

         import numpy as np
         from sklearn.datasets import load_iris
         from sklearn.model_selection import train_test_split
         from sklearn.neighbors import KNeighborsClassifier
         from sklearn.metrics import accuracy_score

         # Load the iris dataset
         iris = load_iris()
         X, y = iris.data, iris.target

         # Split the data into training and testing sets
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

         # Create and fit the model
         knn = KNeighborsClassifier(n_neighbors=3)
         knn.fit(X_train, y_train)

         # Make predictions
         y_pred = knn.predict(X_test)

         # Calculate accuracy
         accuracy = accuracy_score(y_test, y_pred)
         print(f'Accuracy: {accuracy:.2f}')

   .. tab:: Example 3: Decision Tree

      .. code-block:: python

         from sklearn.datasets import load_iris
         from sklearn.model_selection import train_test_split
         from sklearn.tree import DecisionTreeClassifier
         from sklearn.metrics import accuracy_score
         from sklearn import tree
         import matplotlib.pyplot as plt

         # Load the iris dataset
         iris = load_iris()
         X, y = iris.data, iris.target

         # Split the data into training and testing sets
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

         # Create and fit the model
         clf = DecisionTreeClassifier()
         clf.fit(X_train, y_train)

         # Make predictions
         y_pred = clf.predict(X_test)

         # Calculate accuracy
         accuracy = accuracy_score(y_test, y_pred)
         print(f'Accuracy: {accuracy:.2f}')

         # Plot the decision tree
         plt.figure(figsize=(20,10))
         tree.plot_tree(clf, filled=True)
         plt.show()


.. challenge::

   Try running ``titanic_sklearn.ipynb`` that can be found in ``Exercises/examples/programs`` directory, on an interactive CPU node. Copy the ``.ipynb`` file into your personal folder. Also copy the ``data`` directory into your personal folder as it contains the dataset for this and subsequent Exercises.

   Run it on a jupyter notebook on an interactive CPU node. An interative GPU node will also do. 

   Load the correct modules that contain scikit-learn, numpy, seaborn, pandas, matplotlib and jupyter libraries before starting the jupyter notebook. Users on NSC can use prebuilt ``tf_env`` or ``torch_env`` venv.

   * Learning outcomes:
      - How to load a jupyter notebook on an interactive node.
      - How to load correct modules already available on the system, in order to run scikit-learn.



PyTorch and TensorFlow
-----------------------

The following table demonstrates some common tasks in PyTorch and TensorFlow, highlighting their similarities and differences through code examples:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **PyTorch**
     - **TensorFlow**
   * - 
       .. code-block:: python

          import torch
          import torch.nn as nn
          import torch.optim as optim

          # Tensor creation with gradients enabled
          x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)

          # Automatic differentiation
          y = x.sum()
          y.backward()
          print("Gradient of x:", x.grad)

          # Creating and using a neural network layer
          layer = nn.Linear(2, 2)
          input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
          output = layer(input_tensor)
          print("Layer output:", output)

          # Optimizer usage
          optimizer = optim.SGD(layer.parameters(), lr=0.01)
          loss = output.sum()
          optimizer.zero_grad()  # Clear gradients
          loss.backward()        # Compute gradients
          optimizer.step()       # Update weights
          print("Updated weights:", layer.weight)

     - 
       .. code-block:: python

          import tensorflow as tf
          from tensorflow.keras.layers import Dense
          from tensorflow.keras.optimizers import SGD

          # Tensor creation
          x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)

          # Automatic differentiation
          with tf.GradientTape() as tape:
              y = tf.reduce_sum(x)
          grads = tape.gradient(y, x)
          print("Gradient of x:", grads)

          # Creating and using a neural network layer
          layer = Dense(2)
          input_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)
          output = layer(input_tensor)
          print("Layer output:", output)

          # Optimizer usage
          optimizer = SGD(learning_rate=0.01)
          with tf.GradientTape() as tape:
              loss = tf.reduce_sum(output)
          gradients = tape.gradient(loss, layer.trainable_variables)
          optimizer.apply_gradients(zip(gradients, layer.trainable_variables))
          print("Updated weights:", layer.weights)


We now learn by submitting a batch job which consists of loading python module, activating python environment and running DNN code for image classification.

.. admonition:: Fashion MNIST image classification using Pytorch/TensorFlow
   :class: dropdown

   .. tabs::

      .. tab:: Pytorch

         .. code-block:: python
            
            import torch
            from torch import nn
            from torch.utils.data import DataLoader
            from torchvision import datasets
            from torchvision.transforms import ToTensor
   
            # Load FashionMNIST data
            training_data = datasets.FashionMNIST(
               root="data/pytorch",
               train=True,
               download=False,
               transform=ToTensor(),
            )
   
            test_data = datasets.FashionMNIST(
               root="data/pytorch",
               train=False,
               download=False,
               transform=ToTensor(),
            )
   
            batch_size = 32
   
            # Create data loaders.
            train_dataloader = DataLoader(training_data, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
   
            for X, y in test_dataloader:
               print(f"Shape of X [N, C, H, W]: {X.shape}")
               print(f"Shape of y: {y.shape} {y.dtype}")
               break
               
            # Define device
            device = (
               "cuda"
               if torch.cuda.is_available()
               else "cpu"
            )
   
            print(f"Using {device} device")
   
            # Define model
            class NeuralNetwork(nn.Module):
               def __init__(self):
                  super().__init__()
                  self.flatten = nn.Flatten()
                  self.linear_relu_stack = nn.Sequential(
                        nn.Linear(28*28, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                  )
   
               def forward(self, x):
                  x = self.flatten(x)
                  logits = self.linear_relu_stack(x)
                  return logits
   
            model = NeuralNetwork().to(device)
   
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
            # Train and evaluate the model
            def train(dataloader, model, loss_fn, optimizer):
               size = len(dataloader.dataset)
               model.train()
               for batch, (X, y) in enumerate(dataloader):
                  X, y = X.to(device), y.to(device)
   
                  # Compute prediction error
                  pred = model(X)
                  loss = loss_fn(pred, y)
   
                  # Backpropagation
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
   
                  if batch % 100 == 0:
                        loss, current = loss.item(), (batch + 1) * len(X)
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                        
            def test(dataloader, model, loss_fn):
               size = len(dataloader.dataset)
               num_batches = len(dataloader)
               model.eval()
               test_loss, correct = 0, 0
               with torch.no_grad():
                  for X, y in dataloader:
                        X, y = X.to(device), y.to(device)
                        pred = model(X)
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
               test_loss /= num_batches
               correct /= size
               print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
               
            epochs = 10
            for t in range(epochs):
               print(f"Epoch {t+1}\n-------------------------------")
               train(train_dataloader, model, loss_fn, optimizer)
               test(test_dataloader, model, loss_fn)
            print("Done!")
   
            # Class names for FashionMNIST
            classes = [
               "T-shirt/top",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle boot",
            ]
   
            model.eval()
   
            # Predict and display results for one example
            x, y = test_data[0][0], test_data[0][1]
            with torch.no_grad():
               x = x.to(device)
               pred = model(x)
               predicted, actual = classes[pred[0].argmax(0)], classes[y]
               print(f'Predicted: "{predicted}", Actual: "{actual}"')
               
      .. tab:: TensorFlow
         
         .. code-block:: python

            import tensorflow as tf
            import numpy as np
            from utils import load_data_fromlocalpath
            
            # Load FashionMNIST data
            (train_images, train_labels), (test_images, test_labels) = load_data_fromlocalpath("data/tf")
               
            # Define device
            device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
            print(f"Using {device} device")
   
            # Define the model
            class NeuralNetwork(tf.keras.Model):
               def __init__(self):
                  super(NeuralNetwork, self).__init__()
                  self.flatten = tf.keras.layers.Flatten()
                  self.dense1 = tf.keras.layers.Dense(128, activation='relu')
                  self.dense2 = tf.keras.layers.Dense(128, activation='relu')
                  self.dense3 = tf.keras.layers.Dense(10)
   
               def call(self, x):
                  x = self.flatten(x)
                  x = self.dense1(x)
                  x = self.dense2(x)
                  return self.dense3(x)
   
            model = NeuralNetwork()
               
            model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
   
   
            # Train and evaluate the model      
            model.fit(train_images, train_labels, epochs=10)
   
            test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
   
            print('\nTest accuracy:', test_acc)
   
            # Class names for FashionMNIST
            classes = [
               "T-shirt/top",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle boot",
            ]
   
            # Predict and display results for one example
            probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
   
            # Grab an image from the test dataset.
            x, y = test_images[1], test_labels[1]
   
            # Add the image to a batch where it's the only member.
            x = (np.expand_dims(x,0))
            predictions_single = probability_model.predict(x)
            predicted, actual = classes[np.argmax(predictions_single[0])], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

      .. tab:: utils.py

         .. code-block:: python

            import os
            import numpy as np
            import gzip

            def load_data_fromlocalpath(input_path):
               """Loads the Fashion-MNIST dataset.
               Author: Henry Huang in 2020/12/24.
               We assume that the input_path should in a correct path address format.
               We also assume that potential users put all the four files in the path.

               Load local data from path ‘input_path’.

               Returns:
                     Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
               """
               files = [
                     'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
               ]

               paths = []
               for fname in files:
                  paths.append(os.path.join(input_path, fname))  # The location of the dataset.


               with gzip.open(paths[0], 'rb') as lbpath:
                  y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

               with gzip.open(paths[1], 'rb') as imgpath:
                  x_train = np.frombuffer(
                     imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

               with gzip.open(paths[2], 'rb') as lbpath:
                  y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

               with gzip.open(paths[3], 'rb') as imgpath:
                  x_test = np.frombuffer(
                     imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

               return (x_train, y_train), (x_test, y_test)

.. admonition:: Batch scripts for running image classification using Pytorch/TensorFlow
   :class: dropdown
      
   .. tabs::

      .. tab:: UPPMAX

         .. code-block:: bash 

            #!/bin/bash -l
            #SBATCH -A naiss2024-22-1442 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -p node
            #SBATCH -n 1 # Asking for 1 node
            #SBATCH -M snowy
            #SBATCH --gres=gpu:1 # Asking for 1 GPU

            # Load any modules you need, here Python 3.11.8.
            module load python/3.11.8

            source ../torch_env/bin/activate
            #source ../tf_env/bin/activate #unncomment this for tf env and comment torch env

            # Run your Python script
            python test_pytorch_nn.py

      .. tab:: HPC2N

         .. code-block:: bash 

            #!/bin/bash                                                                     
            #SBATCH -A hpc2n2024-142 # Change to your own                                   
            #SBATCH --time=00:10:00 # Asking for 10 minutes                                 
            #SBATCH -n 1 # Asking for 1 core                                                
            #SBATCH --gpus=1                                                                
            #SBATCH -C nvidia_gpu                                                           

            # Load any modules you need, here for Python/3.11.3
            module load GCC/12.3.0 Python/3.11.3

            source ../torch_env/bin/activate
            #source ../tf_env/bin/activate #unncomment this for tf env and comment torch env

            # Run your Python script                                                        
            python fashion_mnist.py


      .. tab:: LUNARC

            .. code-block:: bash
               
               #!/bin/bash
               #SBATCH -A lu2024-2-88
               #SBATCH -p gpua100
               #SBATCH -n 1
               #SBATCH --ntasks-per-node=1
               #SBATCH -t 0:10:00
               #SBATCH --gres=gpu:1


               # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle
               module load GCC/13.2.0 Python/3.11.5 

               source ../torch_env/bin/activate
               #source ../tf_env/bin/activate #unncomment this for tf env and comment torch env

               # Run your Python script
               python fashion_mnist.py


      .. tab:: NSC      
            
            .. code-block:: bash 
   
               #!/bin/bash
               #SBATCH -A naiss2024-22-1493 # Change to your own
               #SBATCH -n 1
               #SBATCH -c 32
               #SBATCH -t 00:10:00 # Asking for 10 minutes
               #SBATCH --gpus-per-task=1

               ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
               ml load Python/3.11.5

               source ../torch_env/bin/activate
               #source ../tf_env/bin/activate #unncomment this for tf env and comment torch env

               python fashion_mnist.py


.. challenge::

   Try and run the either pytorch or tensorflow code for Fasion MNIST dataset by submitting a batch job.
   The dataset is stored in ``data/pytorch`` or ``data/tf`` directory. Copy the ``data`` directory to your personal folder.
   In order to run this at any HPC resource you should either do a batch job or run interactively on compute nodes. Remember, you should not run long/resource heavy jobs on the login nodes, and they also do not have GPUs if you want to use that.  

   * Learning outcomes:
      - How to submit a batch job on a HPC GPU resource inside a virtual env.
      - How to load the correct modules and activate the correct environment for running PyTorch or TensorFlow code.


Miscellaneous examples
-----------------------


.. admonition:: Running several jobs from within one job
   :class: dropdown

      You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes. 

      This example shows how you would run several programs or variations of programs sequentially within the same job: 

      .. tabs::

         .. tab:: HPC2N

            Example batch script for Kebnekaise, TensorFlow version 2.11.0 and Python version 3.11.3

            .. code-block:: bash 
            
               #!/bin/bash 
               # Remember to change this to your own project ID after the course! 
               #SBATCH -A hpc2n2024-142
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
            
            .. code-block:: bash 

               #!/bin/bash -l
               # Remember to change this to your own project ID after the course!
               #SBATCH -A naiss2024-22-1442
               # We are asking for at least 1 hour
               #SBATCH --time=01:00:01
               #SBATCH -M snowy
               #SBATCH --gres=gpu:1
               #SBATCH --mail-type=begin        # send email when job begins
               #SBATCH --mail-type=end          # send email when job ends
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


         .. tab:: NSC

            Example batch script for Tetralith, TensorFlow version 2.18 and Python version 3.11.5. 
            
            .. code-block:: bash 
   
               #!/bin/bash
               #SBATCH -A naiss2024-22-1493 # Change to your own
               #SBATCH -n 1
               #SBATCH -c 32
               #SBATCH -t 00:10:00 # Asking for 10 minutes
               #SBATCH --gpus-per-task=1

               ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
               ml load Python/3.11.5

               source ../tf_env/bin/activate
               # Output to file - not needed if your job creates output in a file directly
               # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
               python tf_program.py 1 2 > myoutput1 2>&1
               cp myoutput1 mydatadir
               python tf_program.py 3 4 > myoutput2 2>&1
               cp myoutput2 mydatadir
               python tf_program.py 5 6 > myoutput3 2>&1
               cp myoutput3 mydatadir

         .. tab:: LUNARC

            Example batch script for Cosmos, TensorFlow version 2.15 and Python version 3.11.8. 
            
            .. code-block:: bash 

               #!/bin/bash
               #SBATCH -A lu2024-2-88
               #SBATCH -p gpua100
               #SBATCH -n 1
               #SBATCH --ntasks-per-node=1
               #SBATCH -t 0:10:00
               #SBATCH --gres=gpu:1


               # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle
               module load GCC/13.2.0 Python/3.11.5 

               source ../torch_env/bin/activate
               #source ../tf_env/bin/activate #unncomment this for tf env and comment torch env
               
               # Output to file - not needed if your job creates output in a file directly
               # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
               python tf_program.py 1 2 > myoutput1 2>&1
               cp myoutput1 mydatadir
               python tf_program.py 3 4 > myoutput2 2>&1
               cp myoutput2 mydatadir
               python tf_program.py 5 6 > myoutput3 2>&1
               cp myoutput3 mydatadir

.. admonition:: Scikit-Learn + TensorFlow using modules 
    :class: dropdown

      .. code-block:: python 
        
         # fit a final model and make predictions on new data for the ionosphere dataset
         from pandas import read_csv
         from sklearn.preprocessing import LabelEncoder
         from sklearn.metrics import accuracy_score
         from tensorflow.keras import Sequential
         from tensorflow.keras.layers import Dense
         from tensorflow.keras.layers import Dropout
         # load the dataset
         path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
         df = read_csv(path, header=None)
         # split into input and output columns
         X, y = df.values[:, :-1], df.values[:, -1]
         # ensure all data are floating point values
         X = X.astype('float32')
         # encode strings to integer
         le = LabelEncoder()
         y = le.fit_transform(y)
         # determine the number of input features
         n_features = X.shape[1]
         # define model
         model = Sequential()
         model.add(Dense(50, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
         model.add(Dropout(0.4))
         model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
         model.add(Dropout(0.4))
         model.add(Dense(1, activation='sigmoid'))
         # compile the model
         model.compile(optimizer='adam', loss='binary_crossentropy')
         # fit the model
         model.fit(X, y, epochs=100, batch_size=8, verbose=0)
         # define a row of new data
         row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
         # make prediction
         #for tf>2.6 uncomment the following line but comment the next line
         #yhat = model.predict(x=np.array([row]))
         yhat = model.predict_classes([row]) 
         # invert transform to get label for class
         yhat = le.inverse_transform(yhat)
         # report prediction
         print('Predicted: %s' % (yhat[0]))


      .. tabs::

         .. tab:: HPC2N
         
            .. code-block:: bash 
            
                  #!/bin/bash 
                  # Remember to change this to your own project ID after the course! 
                  #SBATCH -A hpc2n2024-142
                  # We are asking for 5 minutes
                  #SBATCH --time=00:05:00
                  # Asking for one V100
                  #SBATCH --gres=gpu:v100:1
                  
                  # Remove any loaded modules and load the ones we need
                  module purge  > /dev/null 2>&1
                  module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3 scikit-learn/1.4.2

                  # Run your Python script 
                  python example-tf.py 
                  
         .. tab:: UPPMAX
         
            .. code-block:: bash 
            
                  #!/bin/bash -l  
                  # Remember to change this to your own project ID after the course! 
                  #SBATCH -A naiss2024-22-1442
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


         .. tab:: NSC

            .. code-block:: bash 

               #!/bin/bash
               #SBATCH -A naiss2024-22-1493 # Change to your own
               #SBATCH -n 1
               #SBATCH -c 32
               #SBATCH -t 00:10:00 # Asking for 10 minutes
               #SBATCH --gpus-per-task=1

               ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
               ml load Python/3.11.5

               source ../tf_env/bin/activate
               
               # Run your Python script 
               python example-tf.py 


         .. tab:: LUNARC
            
            .. code-block:: bash 

               #!/bin/bash
               #SBATCH -A lu2024-2-88
               #SBATCH -p gpua100
               #SBATCH -n 1
               #SBATCH --ntasks-per-node=1
               #SBATCH -t 0:10:00
               #SBATCH --gres=gpu:1


               # Load any modules you need, here for Python/3.10.4 and compatible SciPy-bundle
               module load GCC/11.3.0 Python/3.10.4 SciPy-bundle/2022.05 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2

               
               # Run your Python script 
               python example-tf.py 





Exercises
---------

.. challenge::

   Try running a pytorch code for fitting a third degree polynomial to a sine function. Use the pytorch provided by module systems instead of using the virtual environment (except if you are on Tetralith (NSC), there is no pytorch available).
   Submit the job using either a batch script or run the code interactively on a GPU node (if you already are on one).

   Visit the `List of installed ML/DL tools <#list-of-installed-ml-dl-tools>`_ and make sure to load the correct pre-requisite modules like correct python version and GCC if needed.

   .. admonition:: Fit a third order polynomial to a sine function.
    :class: dropdown

        The below program can be found in the ``Exercises/examples/programs`` directory under the name ``pytorch_fitting_gpu.py``. 

        .. code-block:: python
        
            # source : https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors
            
            import torch
            import math
            
            dtype = torch.float
            #device = torch.device("cpu")
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

   .. admonition:: Output via an interactive Snowy session
    :class: dropdown

        .. code-block:: bash

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


   * Learning outcomes:
      - How to load pytorch/tensorflow from module system instead of using virtual environment.
      - Run the job on a GPU node either interactively or via batch script.


.. keypoints::

  - At all clusters you will find PyTorch, TensorFlow, Scikit-learn under different modules, except Tetralith (NSC). 
  - When in doubt, search your modules and its correct version using ``module spider``.  If you still wished to have the correct versions for each cluster, check the `summary page <https://uppmax.github.io/HPC-python/summary2.html#summary-day2>`_.
  - If you plan to use mutiple libraries with complex dependencies, it is recommended to use a virtual environment and pip install your libraries.
  - Always run heavy ML/DL jobs on compute nodes and not on login nodes. For development purpose, you can use an interactive session on a compute node.

