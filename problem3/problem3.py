# Standard Libraries
import os
import numpy as np

# Load .mat files
import scipy.io

def HOSVD(
    X : np.ndarray,
    R1 : int = 30,
    R2 : int = 30,
    R3 : int = 20,
    tolerance : float = 1e-4
) -> tuple[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
    I, J, K = X.shape

    G = np.random.rand(R1, R2, R3)

    A = np.random.rand(I, R1)
    B = np.random.rand(J, R2)
    C = np.random.rand(K, R3)

    convergence = False

    factors = [A, B, C]
    dimensions = [R1, R2, R3]
    while not convergence:
        G_previous = G.copy()

        for _k in range(len(factors)):
            y = np.tensordot(X, factor[0].T, axes=([1], [0])) # First mode
            y = np.tensordot(y, factor[1].T, axes=([2], [0])) # Second mode
            y = np.tensordot(y, factor[2].T, axes=([2], [1])) # Third mode

            u, s, vh = np.linalg.svd(y, full_matrices = False)

            factor[_k] = s[:dimensions[_k]]

        G = np.tensordot(X, A.T, axes=([1], [0])) # First mode
        G = np.tensordot(G, B.T, axes=([2], [0])) # Second mode
        G = np.tensordot(G, C.T, axes=([2], [1])) # Third mode

        is_close = np.allclose(G, G_previous, rtol = tolerance, atol = 0)
        convergence = True if is_close else False
    
    return (G, (A1, A2, A3))

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
    tensor_frobenius_norm = np.linalg.norm(tensor, 'fro')
    core_frobenius_norm = np.linalg.norm(core, 'fro')

    numerator = np.sqrt(tensor_frobenius_norm**2 - core_frobenius_norm**2)
    denominator = tensor_frobenius_norm

    relative_error = numerator / denominator

    return relative_error

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)

    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Problem3', 'ATT.mat'))
    _a = scipy.io.loadmat(file_directory)['A']

    core, factors = HOSVD(_a)

    print(core.shape)