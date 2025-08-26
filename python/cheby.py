import numpy as np
import numpy.polynomial.chebyshev as cheb


def cheby_eval_1d(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluates a 1D Chebyshev series using Clenshaw's algorithm."""
    x = np.atleast_1d(x)
    n = len(coeffs)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.full_like(x, coeffs[0])
    
    b_k_plus_2 = np.zeros_like(x)
    b_k_plus_1 = np.zeros_like(x)
    x2 = 2 * x
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + x2 * b_k_plus_1 - b_k_plus_2
        b_k_plus_2, b_k_plus_1 = b_k_plus_1, b_k
    y = coeffs[0] + x * b_k_plus_1 - b_k_plus_2
    return y

def cheby_5d_evaluate(coeffs: np.ndarray, p_vecs: list) -> np.ndarray:
    """
    Evaluates a 5D Chebyshev series using np.einsum for simplicity and speed.
    
    Args:
        coeffs: The 5D tensor of Chebyshev coefficients.
        p_vecs: A list of 5 1D arrays for the coordinates along each axis.
    """
    eval_matrices = [
        cheb.chebvander(p, deg)
        for p, deg in zip(p_vecs, (d - 1 for d in coeffs.shape))
    ]

    return np.einsum(
        'abcde,ia,jb,kc,ld,me->ijklm',
        coeffs,
        *eval_matrices
    )

def evaluate_cheb_poly_mat(R: np.ndarray, z: float) -> np.ndarray:
    """Evaluates the matrix polynomial R(z) = sum(R_k * T_k(z))."""
    n_z = R.shape[2]
    B_k_plus_2 = np.zeros_like(R[:, :, 0], dtype=np.complex128)
    B_k_plus_1 = np.zeros_like(R[:, :, 0], dtype=np.complex128)
    z2 = 2 * z
    for k in range(n_z - 1, 0, -1):
        B_k = R[:, :, k] + z2 * B_k_plus_1 - B_k_plus_2
        B_k_plus_2, B_k_plus_1 = B_k_plus_1, B_k

    Rz = R[:, :, 0] + z * B_k_plus_1 - B_k_plus_2
    return Rz