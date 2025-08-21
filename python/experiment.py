import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from rootfinder import roots_z
from jac import get_jacobian_func_numeric

# --- Experiment Configuration ---
PLOT_DIST = True
DEGREE_APPROX = 2
SHOULD_PRECONDITION = True

# System parameters
del_param = 1e-7

# Subregion widths
h_vals = np.logspace(-6, 4, 80)

# Iteration counts
NUM_Q = 1
NUM_ROOT_LOC = 50

# --- Function Definition ---
# Standard NumPy/Python functions
f1 = lambda x, y, z: (1/3 + del_param) * x + 1/3 * y + (1/3 - del_param) * z
f2 = lambda x, y, z: (1/3 + del_param) * x + (1/3 - del_param) * y + 1/3 * z
f3 = lambda x, y, z: 1/3 * x + (1/3 + del_param) * y + (1/3 - del_param) * z
functions = [f1, f2, f3]

# --- Data Storage ---
dist_vals = np.full((len(h_vals), NUM_Q, NUM_ROOT_LOC), np.nan)
predicted_dist_vals = np.full((len(h_vals), NUM_Q, NUM_ROOT_LOC), np.nan)

# --- Main Loop ---
for i_Q in range(NUM_Q):
    J_func = get_jacobian_func_numeric(functions) # <-- Use the numeric version
    J = J_func(0.0, 0.0, 0.0)
    J_inv = np.linalg.inv(J)
    cond_root = np.linalg.norm(J_inv)
    cond_eig = 1.0 / np.abs(np.linalg.det(J))

    progress_bar = tqdm(
        range(NUM_ROOT_LOC),
        desc=f"Q #{i_Q+1}/{NUM_Q}",
        leave=False
    )
    for i_root_loc in progress_bar:
        expected_root = (2 * np.random.rand(3) - 1) / 100.0

        for k, h in enumerate(h_vals):
            a = expected_root - h / 2
            b = expected_root + h / 2
            cube_scale = (b - a) / 2
            cube_shift = (b + a) / 2

            def remap(coords, idx):
                return cube_scale[idx] * coords + cube_shift[idx]

            f_remapped = [
                lambda x,y,z, f=f: f(remap(x,0), remap(y,1), remap(z,2))
                for f in functions
            ]

            p = f_remapped

            if SHOULD_PRECONDITION:
                J_func_transformed = get_jacobian_func_numeric(p)
                J_transformed = J_func_transformed(0.0, 0.0, 0.0)
                
                # Avoid singular matrix errors
                if np.linalg.cond(J_transformed) > 1/np.finfo(float).eps:
                    continue
                J_inv_transformed = np.linalg.inv(J_transformed)

                p_tmp = p[:]
                p = [
                    lambda x,y,z, i=i: sum(J_inv_transformed[i, j] * p_tmp[j](x, y, z) for j in range(3))
                    for i in range(3)
                ]

            # Solve in the unit cube
            roots_z_unit, _, _, _ = roots_z(p, DEGREE_APPROX)

            if roots_z_unit.size == 0:
                continue

            roots_z_physical = remap(roots_z_unit, 2)
            dists = np.abs(roots_z_physical - expected_root[2])
            min_dist_idx = np.argmin(dists)
            dist_vals[k, i_Q, i_root_loc] = dists[min_dist_idx]

# --- Plotting ---
if PLOT_DIST:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_dist = np.nanmean(dist_vals, axis=(1, 2))

    ax.loglog(h_vals, mean_dist, 'o-', label=r'Observed Error $|z - z_*|$')

    ax.axhline(cond_root * 1e-16, color='r', linestyle='--', label=r'Error Floor $\approx u \cdot \kappa_{root}$')
    ax.axhline(cond_eig * 1e-16, color='g', linestyle='-.', label=r'Error Floor $\approx u \cdot \kappa_{eig}$')

    ax.set_xlabel('Box Width $h$')
    ax.set_ylabel('Error in $z$-component')
    ax.set_title(f'Effect of Domain Shrinking (δ = {del_param:.0e})')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()