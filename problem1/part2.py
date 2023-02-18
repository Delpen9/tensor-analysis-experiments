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