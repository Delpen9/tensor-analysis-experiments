# Standard Libraries
import os
import numpy as np

# Load .mat files
import scipy.io

def HOSVD(
    X : np.ndarray,
    R1 : int = 30,
    R2 : int = 30,
    R3 : int = 20
) -> tuple[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
    # Initialize the core tensor G
    G = np.copy(X)
    
    # Initialize the left matrices A1, A2, A3
    A1 = np.zeros((X.shape[0], R1))
    A2 = np.zeros((X.shape[1], R2))
    A3 = np.zeros((X.shape[2], R3))
    
    # Update the left matrix A1
    for i in range(X.shape[0]):
        G[i, :, :] = np.dot(A1[i, :], np.dot(A2, A3.T))
    U, S, VT = np.linalg.svd(np.reshape(G, (X.shape[0], -1)), full_matrices=False)
    A1 = U[:, :R1]
    G = np.dot(np.diag(S[:R1]), VT[:R1, :]).reshape(G.shape)
    
    # Update the left matrix A2
    for i in range(X.shape[1]):
        G[:, i, :] = np.dot(A2[i, :], np.dot(A1, A3.T))
    U, S, VT = np.linalg.svd(np.reshape(G, (X.shape[1], -1)), full_matrices=False)
    A2 = U[:, :R2]
    G = np.dot(np.diag(S[:R2]), VT[:R2, :]).reshape(G.shape)
    
    # Update the left matrix A3
    for i in range(X.shape[2]):
        G[:, :, i] = np.dot(A3[i, :], np.dot(A1, A2.T))
    U, S, VT = np.linalg.svd(np.reshape(G, (X.shape[2], -1)), full_matrices=False)
    A3 = U[:, :R3]
    G = np.dot(np.diag(S[:R3]), VT[:R3, :]).reshape(G.shape)
    
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