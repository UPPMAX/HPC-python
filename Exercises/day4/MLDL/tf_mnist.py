import tensorflow as tf
import numpy as np
from utils import load_data_fromlocalpath

# Load FashionMNIST data
(train_images, train_labels), (test_images, test_labels) = load_data_fromlocalpath("datasets/tf")

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