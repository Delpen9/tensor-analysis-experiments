# Standard Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matrix math
import tensorly
import scipy

# Load .mat files
import scipy.io

def hard_thresholding(
    core : np.ndarray,
    percentile : float
):
    '''
    Threshold a tensor given a percentile.

    Parameters
    ----------
        core (np.ndarray):
            A numpy array representing the core tensor.
        percentile (float):
            A value between 0 and 1 representing the percentage
            of coefficients to keep.

    Returns
    -------
        thresholded_core (np.ndarray):
            A NumPy array representing the core tensor after
            hard thresholding.
    '''
    threshold_value = np.percentile(np.abs(core.flatten()), (1 - percentile) * 100)
    thresholded_core = np.where(np.abs(core) >= threshold_value, core, 0)

    return thresholded_core

def relative_error(
    tensor : np.ndarray,
    core : np.ndarray
) -> float:
    '''
    Calculate the relative error between a tensor and its core tensor.

    Parameters
    ----------
    tensor : numpy.ndarray
        The original tensor.
    core : numpy.ndarray
        The core tensor obtained from decomposition.

    Returns
    -------
    float
        The relative error between the original tensor and its core tensor.

    '''
    assert tensor.ndim <= 3 and core.ndim <= 3

    tensor_to_norm = tensor.reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2])).copy()
    core_to_norm = core.reshape((core.shape[0], core.shape[1] * core.shape[2])).copy()

    tensor_frobenius_norm = np.linalg.norm(tensor_to_norm, 'fro')
    core_frobenius_norm = np.linalg.norm(core_to_norm, 'fro')

    numerator = np.sqrt(tensor_frobenius_norm**2 - core_frobenius_norm**2)
    denominator = tensor_frobenius_norm

    relative_error = numerator / denominator

    return relative_error

def save_images_to_directory(
    tensor : np.ndarray,
    directory : str,
    image_names : str
) -> None:
    '''
    Save images at indices 5, 10, and 15 for a given 3-dimensional numpy tensor to a given directory.

    Parameters:
    tensor (numpy.ndarray):
        The 3-dimensional numpy tensor containing the images.
    directory (str):
        The path to the directory where the images should be saved.
    image_names (str):
        Unique name given for current run.
    '''
    image_5 = tensor[:, :, 5]
    image_10 = tensor[:, :, 10]
    image_15 = tensor[:, :, 15]

    plt.imsave(os.path.join(directory, fr'{image_names}_5.png'), image_5)
    plt.imsave(os.path.join(directory, fr'{image_names}_10.png'), image_10)
    plt.imsave(os.path.join(directory, fr'{image_names}_15.png'), image_15)

def HOSVD(
    X : np.ndarray,
    R1 : int = 30,
    R2 : int = 30,
    R3 : int = 20,
    tolerance : float = 1e-4,
    max_iterations : int = 100
) -> tuple[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
    I, J, K = X.shape

    G = np.zeros(shape = (R1, R2, R3), dtype = float)

    A = np.random.rand(I, R1).astype(float)
    B = np.random.rand(J, R2).astype(float)
    C = np.random.rand(K, R3).astype(float)

    convergence = False
    max_iteration_achieved = False

    factors = [A, B, C]
    core_dimensions = [R1, R2, R3]
    tensor_dimensions = [I, J, K]

    iteration = 1
    while not convergence and not max_iteration_achieved:
        G_previous = G.copy()

        for _k in range(len(factors)):
            y = tensorly.tenalg.multi_mode_dot(X.copy(), factors, skip = _k, modes = [0, 1, 2], transpose = True)           

            mode_k_matrix = tensorly.base.unfold(y, mode = _k)

            U, _, _ = np.linalg.svd(mode_k_matrix)

            leading_left_singular_values = U[:, :core_dimensions[_k]]
            factors[_k] = leading_left_singular_values

        G = tensorly.tenalg.multi_mode_dot(X.copy(), factors, transpose = True)

        is_close = np.sum(G) - np.sum(G_previous) < tolerance
        convergence = True if is_close else False

        max_iteration_achieved = True if iteration == max_iterations else False

        iteration += 1
    
    return (G, tuple(factors))

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem3', 'ATT.mat'))
    _a = scipy.io.loadmat(file_directory)['A'].astype(float)

    core, factors = HOSVD(_a)

    relative_error_value = relative_error(_a, core)
    text_file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem3'))
    
    with open(fr'{text_file_directory}/problem3.txt', 'w') as filewriter:
        filewriter.write(f'The relative error for the core tensor is: {relative_error_value}.\n')

    threshold_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem3'))

    percentiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    relative_errors = []
    for percentile in percentiles:
        thresholded_core = hard_thresholding(core, percentile)
        save_images_to_directory(thresholded_core, threshold_directory, fr'percentile_{str(percentile)[0] + str(percentile)[2]}_image')

        relative_error_value = relative_error(_a, thresholded_core)
        relative_errors.append(relative_error_value)
    relative_errors = np.array(relative_errors)

    df_percentile = pd.DataFrame(np.vstack((percentiles, relative_errors)).T, columns = ['percentiles', 'relative_errors'])

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem3', 'percentiles.csv'))
    df_percentile.to_csv(file_directory, index = False)