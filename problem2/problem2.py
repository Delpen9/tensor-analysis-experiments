# Standard Libraries
import os
import numpy as np

# Load .mat files
import scipy.io

# Plotting
import cv2
import matplotlib.pyplot
import seaborn

# Decomposition
from tensorly.decomposition import non_negative_tucker_hals

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem2', 'data', 'matlab', 'binary_mnist_train.mat'))
    binary_mnist_train = scipy.io.loadmat(file_directory)

    _x = binary_mnist_train['X']
    _y = np.array([[0] * 50 + [1] * 50]).T

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem2', 'data', 'matlab', 'binary_mnist_test.mat'))
    binary_mnist_test = scipy.io.loadmat(file_directory)

    x_test = binary_mnist_test['X_test']
    y_test = np.array([[0] * 10 + [1] * 10]).T

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem2', 'data', 'matlab', 'binary_mnist_noisy.mat'))
    binary_mnist_noisy = scipy.io.loadmat(file_directory)

    x_noisy = binary_mnist_noisy['X_noisy']
    y_noisy = np.array([[0] * 10 + [1] * 10]).T

    # Part 1
    _x_reshaped = _x.reshape((100, 28, 28)).copy()
    _x_reshaped_normalized = _x_reshaped / 255.0

    x_test_reshaped = x_test.reshape((20, 28, 28)).copy()
    x_test_reshaped_normalized = x_test_reshaped / 255.0

    x_noisy_reshaped = x_noisy.reshape((20, 28, 28)).copy()
    x_noisy_reshaped_normalized = x_noisy_reshaped / 255.0

    _X_reshaped_0_label = _x_reshaped[:50, :, :].copy()
    _X_reshaped_1_label = _x_reshaped[50:, :, :].copy()

    vector_encoding_0_label = np.mean(_X_reshaped_0_label, axis = 0).astype(np.uint8)
    vector_encoding_1_label = np.mean(_X_reshaped_1_label, axis = 0).astype(np.uint8)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem2', 'part_1_vector_image_label_0.png'))
    cv2.imwrite(file_directory, vector_encoding_0_label)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem2', 'part_1_vector_image_label_1.png'))
    cv2.imwrite(file_directory, vector_encoding_1_label)

    # Part 2
    x_train_core, factors = non_negative_tucker_hals(_x_reshaped_normalized, rank = [100, 20, 20], tol = 1e-4, n_iter_max = 50)

    x_train = x_train_core.reshape(-1, 400).copy()
    y_train = _y.copy()

    clf = RandomForestClassifier(n_estimators = 500, random_state = 1)
    clf.fit(x_train, y_train.flatten())

    y_pred = clf.predict(x_train)

    accuracy = accuracy_score(y_train, y_pred)
    print(fr'Accuracy on the training data: {accuracy}.')

    # Part 3.a
    x_test_core, factors = non_negative_tucker_hals(x_test_reshaped_normalized, rank = [20, 20, 20], tol = 1e-4, n_iter_max = 50)
    x_test = x_test_core.reshape(-1, 400).copy()

    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(fr'Accuracy on the test data: {accuracy}.')

    x_noisy_core, factors = non_negative_tucker_hals(x_noisy_reshaped_normalized, rank = [20, 20, 20], tol = 1e-4, n_iter_max = 50)
    x_noisy = x_noisy_core.reshape(-1, 400).copy()

    y_pred = clf.predict(x_noisy)

    accuracy = accuracy_score(y_noisy, y_pred)
    print(fr'Accuracy on the noisy data: {accuracy}.')

    # Part 3.b

