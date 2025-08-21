import numpy as np
from typing import Callable, List

def get_jacobian_func_numeric(
    f_list: List[Callable[[float, float, float], float]]
) -> Callable[[float, float, float], np.ndarray]:
    """
    Creates a function that computes the Jacobian matrix using numerical
    finite differences.

    Args:
        f_list: A list of three functions [f1, f2, f3].

    Returns:
        A function that takes (x, y, z) and returns the 3x3 Jacobian matrix.
    """
    f1, f2, f3 = f_list
    h = 1e-7  # Small step size for finite difference

    # Define partial derivatives using central differences
    df1_dx = lambda x, y, z: (f1(x + h, y, z) - f1(x - h, y, z)) / (2 * h)
    df1_dy = lambda x, y, z: (f1(x, y + h, z) - f1(x, y - h, z)) / (2 * h)
    df1_dz = lambda x, y, z: (f1(x, y, z + h) - f1(x, y, z - h)) / (2 * h)

    df2_dx = lambda x, y, z: (f2(x + h, y, z) - f2(x - h, y, z)) / (2 * h)
    df2_dy = lambda x, y, z: (f2(x, y + h, z) - f2(x, y - h, z)) / (2 * h)
    df2_dz = lambda x, y, z: (f2(x, y, z + h) - f2(x, y, z - h)) / (2 * h)

    df3_dx = lambda x, y, z: (f3(x + h, y, z) - f3(x - h, y, z)) / (2 * h)
    df3_dy = lambda x, y, z: (f3(x, y + h, z) - f3(x, y - h, z)) / (2 * h)
    df3_dz = lambda x, y, z: (f3(x, y, z + h) - f3(x, y, z - h)) / (2 * h)

    def jacobian_at(x: float, y: float, z: float) -> np.ndarray:
        """Computes the Jacobian matrix at a specific point (x, y, z)."""
        return np.array([
            [df1_dx(x, y, z), df1_dy(x, y, z), df1_dz(x, y, z)],
            [df2_dx(x, y, z), df2_dy(x, y, z), df2_dz(x, y, z)],
            [df3_dx(x, y, z), df3_dy(x, y, z), df3_dz(x, y, z)]
        ])

    return jacobian_at