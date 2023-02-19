# Standard Libraries
import os
import numpy as np

# Decomposition
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

import numpy as np

def part1() -> tuple[str, str, str, str]:
    _lambda = np.array([41.9075, 18.6722])

    _u1 = np.array([
        [0.6981, -0.4153],
        [0.1774, 0.7037],
        [0.6937, -0.5765]
    ])

    _u2 = np.array([
        [0.6192, 0.8159],
        [0.3613, 0.2623],
        [0.6972, 0.5152]
    ])

    _u3 = np.array([
        [0.6212, 0.6793],
        [0.5735, 0.6841],
        [0.5341, 0.2657]
    ])

    _u4 = np.array([
        [0.6693, 0.5947],
        [0.7430, 0.8040]
    ])

    # Part 1 - A
    outer_product_u11_u21 = np.outer(a = _u1[:, 0], b = _u2[:, 0])

    first_outer_product_text = fr'Here is the calculation for the first outer product: {outer_product_u11_u21}.'

    # Part 1 - B
    # The outer product is associative
    rank_1_kruskal = np.outer(a = _u1[:, 0], b = _u2[:, 0])
    rank_1_kruskal = np.outer(a = rank_1_kruskal, b = _u3[:, 0])
    rank_1_kruskal = np.outer(a = rank_1_kruskal, b = _u4[:, 0])
    rank_1_kruskal *= _lambda[0]

    first_term_for_kruskal_tensor_text = fr'The first term for the Kruskal tensor is: {rank_1_kruskal}.'

    rank_2_kruskal = np.outer(a = _u1[:, 1], b = _u2[:, 1])
    rank_2_kruskal = np.outer(a = rank_2_kruskal, b = _u3[:, 1])
    rank_2_kruskal = np.outer(a = rank_2_kruskal, b = _u4[:, 1])
    rank_2_kruskal *= _lambda[1]

    second_term_for_kruskal_tensor_text = fr'The second term for the Kruskal tensor is: {rank_2_kruskal}.'

    kruskal_tensor = rank_1_kruskal + rank_2_kruskal
    calculation_for_kruskal_tensor_text = fr'Here is the calculation for the Kruskal tensor: {kruskal_tensor}.'

    return (
        first_outer_product_text,
        first_term_for_kruskal_tensor_text,
        second_term_for_kruskal_tensor_text,
        calculation_for_kruskal_tensor_text
    )

import numpy as np
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

def part2() -> str:
    _g_11 = np.array([
        [35.2489, 0.7832],
        [0.2884, -4.2162]
    ])

    _g_21 = np.array([
        [0.3406,   2.0238],
        [-1.6475, -2.4122]
    ])

    _g_12 = np.array([
        [0.0883, -1.4479],
        [0.9827,  3.4852]
    ])

    _g_22 = np.array([
        [1.0571, -3.3204],
        [4.3714, -7.8907]
    ])

    core = np.array([
        [_g_11, _g_12],
        [_g_21, _g_22]
    ])

    _u1 = np.array([
        [0.6276, -0.0110],
        [0.5511, -0.7000],
        [0.5499, -0.7141]
    ])

    _u2 = np.array([
        [0.5840,  0.6977],
        [0.3660,  0.2300],
        [0.7246, -0.6785]
    ])

    _u3 = np.array([
        [0.6157, -0.4874],
        [0.5428,  0.8319],
        [0.5712, -0.2651]
    ])

    _u4 = np.array([
        [0.6827,  0.7307],
        [0.7307, -0.6827]
    ])

    tucker_tensor = tucker_to_tensor(
        tucker_tensor = (
            core,
            [_u1, _u2, _u3, _u4]
        )
    )

    output_text = fr'Here is the calculation for the Tucker tensor: {tucker_tensor}.'

    return output_text

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

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Problem1'))

    _text_1, _text_2, _text_3, _text_4 = part1()
    _text_5 = part2()
    _text_6, _text_7 = part3()

    text_list = [_text_1, _text_2, _text_3, _text_4, _text_5, _text_6, _text_7]
    for i in range(len(text_list)):
        with open(fr'{file_directory}/problem1_{i + 1}.txt', 'w') as filewriter:
            filewriter.write(text_list[i])
