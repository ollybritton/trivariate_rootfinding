import numpy as np
from scipy.linalg import eig
import time
from cayley import cayley_resultant

def roots_z(f1, f2, f3, a, b, max_degree):
    """
    Computes the z-coordinates of the roots of a system of three trivariate
    polynomials f1, f2, f3 within a bounding box [a, b].

    This is a Python translation of the MATLAB function `roots_z.m`.

    Args:
        f1 (callable): The first function f1(x, y, z).
        f2 (callable): The second function f2(x, y, z).
        f3 (callable): The third function f3(x, y, z).
        a (list or np.ndarray): The lower bounds of the domain [ax, ay, az].
        b (list or np.ndarray): The upper bounds of the domain [bx, by, bz].
        max_degree (int): The degree of approximation to use for the Cayley resultant.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The computed z-roots.
            - np.ndarray: The Cayley resultant matrix pencil R(z).
    """
    a = np.array(a)
    b = np.array(b)

    scale = (b - a) / 2
    shift = (a + b) / 2

    def remap(val, idx):
        # MATLAB is 1-indexed, Python is 0-indexed.
        return scale[idx] * val + shift[idx]

    f1_hat = lambda x, y, z: f1(remap(x, 0), remap(y, 1), remap(z, 2))
    f2_hat = lambda x, y, z: f2(remap(x, 0), remap(y, 1), remap(z, 2))
    f3_hat = lambda x, y, z: f3(remap(x, 0), remap(y, 1), remap(z, 2))

    print(f'Calculating Cayley resultant using degree: {max_degree}')
    start_time = time.time()
    
    # The Cayley resultant now requires the remapped functions
    R, n_s1, n_s2, _, _, n_z = cayley_resultant(f1_hat, f2_hat, f3_hat, max_degree)
    
    end_time = time.time()
    print(f"Cayley resultant calculation took: {end_time - start_time:.4f} seconds")

    print('Finding where det(R(z)) = 0:')
    start_time = time.time()

    # If all the coefficients of the highest degree are zero, leave them out from the linearization
    z_length = n_z
    for i in range(n_z, 0, -1):                     # i = n_z, …, 1
        if np.linalg.norm(R[:, :, i-1], 'fro') < 1e-12:
            z_length = i - 1                        # <- i‑1, not i
        else:
            break
    
    rts = np.array([])
    if z_length == 0:
        rts = np.array([])
    elif z_length <= 2:
        if z_length == 1:
            # This case is not explicitly handled in the original code,
            # it would result in an error. We return no roots.
            rts = np.array([])
        else: # z_length == 2
            eigenvalues, _ = eig(R[:, :, 0], -R[:, :, 1])
            rts = eigenvalues
    else:
        n = n_s1 * n_s2
        size_c = n * (z_length - 1)

        # Compute the linearization matrices C1 and C2 (Companion pencil)
        C1 = -2 * np.eye(size_c)
        C1[-n:, -n:] = 2 * R[:, :, z_length - 1]
        C1[:n, :n] = -np.eye(n)

        C2 = np.zeros((size_c, size_c))
        if z_length > 2:
            C2[:size_c - n, n:] = np.eye(n * (z_length - 2))
            C2[n:, :size_c - n] += np.eye(n * (z_length - 2))
        
        C2[-n:, -2 * n:-n] = -R[:, :, z_length - 1]

        # Compute the last rows of the coefficient matrix C2
        D_list = [R[:, :, i] for i in range(z_length - 1)]
        D = np.hstack(D_list)
        C2[-n:, :] += D

        # Solve the generalized eigenproblem
        eigenvalues, _ = eig(C2, -C1)
        rts = eigenvalues

    # Remap roots back to the original domain for the z-coordinate
    rts = scale[2] * rts + shift[2]

    # Filter for real roots (imaginary part is close to zero)
    # rts = rts[np.isclose(rts.imag, 0)].real

    # keep eigen‑values whose imaginary part is < 1e‑12 (or 1e‑13 …)
    real_mask = np.abs(rts.imag) < 1e-12
    rts = rts[real_mask].real
    
    # Filter for roots within the domain bounds for z
    rts = rts[(rts >= a[2]) & (rts <= b[2])]

    end_time = time.time()
    print(f"Eigenvalue problem and filtering took: {end_time - start_time:.4f} seconds")

    return rts, R
