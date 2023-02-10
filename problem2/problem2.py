# Standard Libraries
import os
import numpy as np

# Load .mat files
import scipy.io

# Plotting
import cv2

def alternating_least_squares(
    training_set : np.ndarray,
    tolerance : float = 1e-04,
    max_iterations : int = 50,
    dimension : int = 20
) -> np.ndarray:
    '''
    alternating_least_squares()
    '''
    decomposition = np.zeros((dimension, dimension, k))

    for i in range(k):
        U, s, V = np.linalg.svd(training_set[:, :, i], full_matrices = False)

        U = U[:, :dimension]
        s = np.diag(s[:dimension])
        V = V[:dimension, :]
        
        decomposition[:, :, i] = U.dot(s).dot(V)

    return decomposition

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
