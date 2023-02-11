import numpy as np

def tucker_decomposition_als(
    tensor : np.ndarray,
    R1 : int = 30,
    R2 : int = 30,
    R3 : int = 20,
    max_iterations : int = 50,
    tolerance : float = 1e-4
) -> tuple[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
    '''
    Perform Tucker decomposition on a tensor using Alternating Least Squares (ALS) algorithm.

    Parameters
    ----------
    tensor : numpy.ndarray
        The input tensor with shape (I, J, K).
    R1 : int
        The desired rank of the first factor matrix, by default 30.
    R2 : int
        The desired rank of the second factor matrix, by default 30.
    R3 : int
        The desired rank of the third factor matrix, by default 20.
    max_iterations : int, optional
        The maximum number of iterations to perform, by default 50.
    tolerance : float, optional
        The tolerance for the residual, by default 1e-4.

    Returns
    -------
    numpy.ndarray
        The core tensor with shape (R1, R2, R3).
    list of numpy.ndarray
        Factor matrices
    '''
    I, J, K = tensor.shape

    A = np.random.rand(I, R1)
    B = np.random.rand(J, R2)
    C = np.random.rand(K, R3)

    for iteration in range(max_iterations):
        for i in range(I):
            A[i, :] = np.linalg.lstsq(np.dot(B, C.T), tensor[i, :, :].flatten(), rcond=None)[0]

        for j in range(J):
            B[j, :] = np.linalg.lstsq(np.dot(A, C.T), tensor[:, j, :].flatten(), rcond=None)[0]

        for k in range(K):
            C[k, :] = np.linalg.lstsq(np.dot(A, B.T), tensor[:, :, k].flatten(), rcond=None)[0]

        reconstructed_tensor = np.einsum('ijk,ik,jk->ijj', tensor, A, B, C)

        residual = np.linalg.norm(tensor - reconstructed_tensor)
        if residual < tolerance:
            break

    core_tensor = np.einsum('ik,jk,kk->ijk', A, B, C)
    return (core_tensor, (A, B, C))

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
    core, factors = tucker_decomposition_als()