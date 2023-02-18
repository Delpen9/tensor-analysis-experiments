# Standard Libraries
import numpy as np

# Decomposition
from tensorly import tucker_to_tensor, kruskal_to_tensor
from tensorly.decomposition import parafac, tucker

def compute_mse(
    original : np.ndarray,
    recomposed : np.ndarray
) -> float:
    '''
    compute_mse():
        Computes the Mean Squared Error (MSE) between the original and recomposed tensors.

        Args:
        original (np.ndarray): The original tensor.
        recomposed (np.ndarray): The recomposed tensor.

        Returns:
        float: The MSE between the original and recomposed tensors.
    '''
    mse = np.mean((original - recomposed) ** 2)
    return mse

def part3() -> tuple[str, str]:
    _x11 = np.array([
        [4, 2, 6],
        [5, 2, 9],
        [6, 7, 2]
    ])

    _x12 = np.array([
        [4, 5, 8],
        [7, 3, 6],
        [2, 4, 9]
    ])

    _x21 = np.array([
        [4, 3, 6],
        [8, 4, 1],
        [4, 2, 9]
    ])

    _x22 = np.array([
        [5, 1, 8],
        [8, 3, 6],
        [3, 5, 1]
    ])

    _x31 = np.array([
        [6, 3, 8],
        [0, 4, 5],
        [3, 2, 7]
    ])

    _x32 = np.array([
        [9, 3, 5],
        [7, 2, 0],
        [4, 2, 9]
    ])

    tensor = np.array([
        [_x11, _x12],
        [_x21, _x22],
        [_x31, _x32]
    ])

    # Tucker
    tucker_core, tucker_factors = tucker(
        tensor,
        rank = [2, 2, 2, 2]
    )

    tucker_tensor_recomposition = tucker_to_tensor(
        tucker_tensor = (
            tucker_core,
            tucker_factors
        )
    )

    tucker_mse = compute_mse(tensor, tucker_tensor_recomposition)

    mean_square_error_tucker_text = fr'The mean-square error for Tucker decomposition is: {tucker_mse}.'

    # CP
    parafac_factors = parafac(tensor, rank = 2)

    parafac_tensor_recomposition = kruskal_to_tensor(parafac_factors)
    parafac_mse = compute_mse(tensor, parafac_tensor_recomposition)

    mean_square_error_cp_text = fr'The mean-square error for CP decomposition is: {parafac_mse}'

    return (mean_square_error_tucker_text, mean_square_error_cp_text)