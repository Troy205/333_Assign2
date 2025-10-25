"""
Headless birefringence simulation with sky blending.
No display or plotting; returns both physical and sky-blended images.
"""

import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib import intensity_from_wavelength, wavelength_to_rgb, linear_image_to_srgb_image


def brewster_angle(n_air: float, n_glass: float) -> float:
    return np.arctan(n_glass / n_air)


def phase_difference(thickness_map, wl: float) -> np.ndarray:
    return 2 * np.pi * 2 * thickness_map / wl


def fp_amplitude(delta_phi: np.ndarray, r: float) -> np.ndarray:
    expi = np.exp(1j * delta_phi)
    numerator = (1 - r**2) * np.exp(1j * delta_phi / 2)
    denominator = (1 - r**2 * expi)
    return numerator / denominator


def birefringence_retardation(delta_n: np.ndarray, thickness_map: np.ndarray, wl: float) -> np.ndarray:
    return 2 * np.pi * delta_n * thickness_map / wl


def radial_phase(R, wl, z=0.5, radial_scale=0.01):
    k = 2 * np.pi / wl
    r_phys = R * radial_scale
    return k * (np.sqrt(r_phys**2 + z**2) - z)


def optical_intensity(
    R,
    theta,
    wl,
    thickness_map,
    r_eff,
    delta_n_map,
    theta_B,
    use_radial=True,
    radial_z=0.5,
    radial_scale=0.01,
):
    """Compute combined optical intensity pattern."""
    if use_radial:
        delta_phi = radial_phase(R, wl, z=radial_z, radial_scale=radial_scale)
    else:
        delta_phi = phase_difference(thickness_map, wl)

    fp_complex = fp_amplitude(delta_phi, r_eff)
    fp_intensity = np.abs(fp_complex)**2

    retard = birefringence_retardation(delta_n_map, thickness_map, wl)
    biref_mod = np.sin(retard / 2)**2
    angular_mod = np.cos(theta - theta_B)**2

    wavelength_weight = intensity_from_wavelength(wl)
    return wavelength_weight * angular_mod * fp_intensity * biref_mod


def make_grid(size=400):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return X, Y, R, theta


def compute_chunk(wavelengths_chunk, R, theta, theta_B, params):
    chunk_img = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.float64)
    for wl in wavelengths_chunk:
        intensity = optical_intensity(
            R=R,
            theta=theta,
            wl=wl,
            thickness_map=params["thickness_map"],
            r_eff=params["r_eff"],
            delta_n_map=params["delta_n_map"],
            theta_B=theta_B,
            use_radial=params.get("use_radial", True),
            radial_z=params.get("radial_z", 0.5),
            radial_scale=params.get("radial_scale", 0.01),
        )
        r_frac, g_frac, b_frac = wavelength_to_rgb(wl * 1e9)
        chunk_img[..., 0] += intensity * r_frac
        chunk_img[..., 1] += intensity * g_frac
        chunk_img[..., 2] += intensity * b_frac
    return chunk_img


def chunk_list(lst, n_chunks):
    lst = list(lst)
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def main(
    size=600,
    wl_min=400e-9,
    wl_max=700e-9,
    n_wavelengths=120,
    base_thickness=1e-6,
    r_eff=0.2,
    delta_n=2e-3,
    n_workers=None,
    radial_z=0.5,
    radial_scale=0.01,
    sky_rgb=np.array([0.45, 0.65, 1.0]),
    alpha=0.6,
):
    """Headless birefringence simulation (returns arrays, no display)."""

    X, Y, R, theta = make_grid(size)
    n_air, n_glass = 1.0, 1.5
    theta_B = brewster_angle(n_air, n_glass)
    wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)

    thickness_map = np.full_like(R, base_thickness)
    delta_n_map = np.full_like(R, delta_n)

    params = dict(
        thickness_map=thickness_map,
        r_eff=r_eff,
        delta_n_map=delta_n_map,
        use_radial=True,
        radial_z=radial_z,
        radial_scale=radial_scale,
    )

    if n_workers is None:
        n_workers = min(32, max(1, (os.cpu_count() or 2)))
    n_workers = min(n_workers, len(wavelengths))
    wl_chunks = chunk_list(wavelengths, n_workers)

    partials = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_chunk, chunk, R, theta, theta_B, params)
                   for chunk in wl_chunks if len(chunk) > 0]
        for future in as_completed(futures):
            partials.append(future.result())

    image = np.sum(partials, axis=0)
    if np.max(image) > 0:
        image /= np.max(image)

    # --- Convert linear -> displayable
    display_img = linear_image_to_srgb_image(image, clip_percentile=100)
    display_img = (1 - alpha) * sky_rgb + alpha * display_img
    display_img = np.clip(display_img, 0, 1)

    # Return both physical and display versions
    return image, display_img


