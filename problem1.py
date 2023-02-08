import numpy as np

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

