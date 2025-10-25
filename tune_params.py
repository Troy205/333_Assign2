"""
auto_fit.py
-----------
Automatic parameter fitting for birefringence simulation
to match a real photograph (real_photo.png).
"""

from parameter_tuning.simulation_headless import main as run_simulation

import numpy as np
from PIL import Image #pips installs with matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import differential_evolution
import os
import time
import multiprocessing as mp


REAL_IMAGE_PATH = "parameter_tuning/real_photo.png"
#OUTPUT_DIR = "fit_results"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

SIM_SIZE = 300
N_WAVELENGTHS = 60

'''During the best attempt we got rgb = (0.33, 0.64, 0.83) and alpha = 0.20.
Hence I have just set these values as there should not be a need to adjust them'''

SKY_RGB = np.array([0.33, 0.64, 0.83])
ALPHA = 0.2

def load_real_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr /= np.percentile(arr, 99)
    return np.clip(arr, 0, 1)


def image_difference(sim_img, real_img):
    score = ssim(sim_img, real_img, channel_axis=-1, data_range=1.0)
    return 1.0 - score


def simulate_display(r_eff, delta_n, radial_z, radial_scale, base_thickness): #If you want to add params to auto-tune, you would add them here
    _, display_img = run_simulation(
        size=SIM_SIZE,
        wl_min=400e-9,
        wl_max=700e-9,
        n_wavelengths=N_WAVELENGTHS,
        base_thickness=base_thickness,
        r_eff=r_eff,
        delta_n=delta_n,
        n_workers=None,
        radial_z=radial_z,
        radial_scale=radial_scale,
        sky_rgb=SKY_RGB,
        alpha=ALPHA,
    )
    return display_img


def objective(params, real_img):
    r_eff, delta_n, radial_z, radial_scale, base_thickness = params
    start = time.time()
    sim_display = simulate_display(r_eff, delta_n, radial_z, radial_scale, base_thickness)
    diff = image_difference(sim_display, real_img)

    elapsed = time.time() - start
    print(
        f"[r_eff={r_eff:.3f}, Δn={delta_n:.4e}, z={radial_z:.3f}, scale={radial_scale:.4f}, "
        f"t={base_thickness:.2e}]"
        f"→ diff={diff:.5f} ({elapsed:.1f}s)"
    )
    return diff


if __name__ == "__main__":
    mp.freeze_support()

    real_img = load_real_image(REAL_IMAGE_PATH, SIM_SIZE)
    '''
    # Parameter bounds
    param_bounds = [
        (0.1, 0.9),        # r_eff
        (1e-4, 5e-3),      # delta_n
        (1.0, 20.0),       # radial_z
        (0.001, 0.02),     # radial_scale
        (1e-7, 5e-6),      # base_thickness (meters)
        (0.2, 0.7),        # sky_r
        (0.4, 0.8),        # sky_g
        (0.7, 1.0),        # sky_b
        (0.2, 0.8),        # alpha
    ]'''

    param_bounds = [
        (0.1, 0.9),        # r_eff
        (1e-4, 5e-3),      # delta_n
        (1.0, 20.0),       # radial_z
        (0.001, 0.1),     # radial_scale
        (0.001, 0.01),      # base_thickness (meters)
    ]

    print("\n🚀 Starting parameter fitting...\n")
    start_global = time.time()

    result = differential_evolution(
        objective,
        bounds=param_bounds,
        args=(real_img,),
        maxiter=15,
        popsize=8,
        workers=-1,
        updating="deferred",
        polish=True,
        disp=True,
    )

    print(f"\n Optimization complete in {(time.time() - start_global)/60:.1f} min")
    print("Best parameters:", result.x)
    print("Best score:", result.fun)

    # Generate final high-res image
    best = result.x
    _, final_img = run_simulation(
        size=800,
        wl_min=400e-9,
        wl_max=700e-9,
        n_wavelengths=120,
        base_thickness=best[4],
        r_eff=best[0],
        delta_n=best[1],
        n_workers=None,
        radial_z=best[2],
        radial_scale=best[3],
        sky_rgb=SKY_RGB,
        alpha=ALPHA,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(real_img)
    plt.title("Real Photo")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(final_img)
    plt.title("Best Simulation Match")
    plt.axis("off")
    plt.tight_layout()
    plt.show()