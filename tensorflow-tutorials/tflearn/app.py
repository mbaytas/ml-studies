import tensorflow as tf
import numpy as np

# Dataset filenames
IRIS_TRAINING = "iris_test.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets
training_set = tf.contrib.learn.datasets.base.load_csv(IRIS_TRAINING, np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(IRIS_TEST, np.int)

x_train = training_set.data
y_train = training_set.target

x_test = test_set.data
y_test = test_set.target

# Build 3-layer DNN w/ 10, 20, 10 units
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10],
                                            n_classes=3)

# Fit model
classifier.fit(x_train, y_train, steps=200)

# Evaluate accuracy
accuracy_score = classifier.evaluate(x_test, y_test)["accuracy"]
print "Accuracy:", accuracy_score

# Classify two new flower samples
new_samples = np.array([[6.4, 3.2, 4.5, 1.5],
                        [5.8, 3.1, 5.0, 1.7]],
                       dtype=float)
y = classifier.predict(new_samples)
print "Predictions:", y