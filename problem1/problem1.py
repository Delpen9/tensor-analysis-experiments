import numpy as np
from tensorly import tucker_to_tensor
from tensorly.decomposition import tucker

if __name__ == '__main__':
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

    print(fr'''
Here is the calculation for the first outer product:
{outer_product_u11_u21}
    ''')

    # Part 1 - B
    # The outer product is associative
    rank_1_kruskal = np.outer(a = _u1[:, 0], b = _u2[:, 0])
    rank_1_kruskal = np.outer(a = rank_1_kruskal, b = _u3[:, 0])
    rank_1_kruskal = np.outer(a = rank_1_kruskal, b = _u4[:, 0])
    rank_1_kruskal *= _lambda[0]

    print(fr'''
The first term for the Kruskal tensor is:
{rank_1_kruskal}
    ''')

    rank_2_kruskal = np.outer(a = _u1[:, 1], b = _u2[:, 1])
    rank_2_kruskal = np.outer(a = rank_2_kruskal, b = _u3[:, 1])
    rank_2_kruskal = np.outer(a = rank_2_kruskal, b = _u4[:, 1])
    rank_2_kruskal *= _lambda[1]

    print(fr'''
The second term for the Kruskal tensor is:
{rank_2_kruskal}
    ''')

    kruskal_tensor = rank_1_kruskal + rank_2_kruskal
    print(fr'''
Here is the calculation for the Kruskal tensor:
{kruskal_tensor}
    ''')

    # Part 2
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

    # Validate the Tucker recomposition
    core, factors = tucker(
        tucker_tensor,
        rank = [2, 2, 2, 2]
    )

    print(fr'''
Here is the calculation for the Tucker tensor:
{tucker_tensor}
    ''')

    # Part 3
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

    core, factors = tucker(
        tucker_tensor
    )

    print(core)