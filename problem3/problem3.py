# Standard Libraries
import os
import numpy as np

# Matrix math
import tensorly
import scipy

# Load .mat files
import scipy.io

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

    A = np.zeros(shape = (I, R1), dtype = float)
    B = np.zeros(shape = (J, R2), dtype = float)
    C = np.zeros(shape = (K, R3), dtype = float)

    convergence = False

    factors = [A, B, C]
    core_dimensions = [R1, R2, R3]
    tensor_dimensions = [I, J, K]

    iteration = 1
    while not convergence:
        G_previous = G.copy()

        for _k in range(len(factors)):
            y = tensorly.tenalg.multi_mode_dot(X.copy(), factors, skip = _k, transpose = True)

            before = _k - 1
            after = (_k + 1) % 3
            y = y.reshape(y.shape[_k], y.shape[before] * y.shape[after])

            u, s, vh = np.linalg.svd(y, full_matrices = False)

            leading_left_singular_values = u[:core_dimensions[_k]].T
            factors[_k] = leading_left_singular_values

        G = tensorly.tenalg.multi_mode_dot(X.copy(), factors, transpose = True)

        is_close = np.allclose(G, G_previous, rtol = tolerance, atol = 0)
        convergence = True if is_close else False

        convergence = True if iteration == 100 else False

        iteration += 1
    
    return (G, tuple(factors))

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem3', 'ATT.mat'))
    _a = scipy.io.loadmat(file_directory)['A']

    core, factors = HOSVD(_a)

    print(_a.shape)

    print(relative_error(_a, core))

    print(core.shape)