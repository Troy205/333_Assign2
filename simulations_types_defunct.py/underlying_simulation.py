import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from lib import intensity_from_wavelength, wavelength_to_rgb


def brewster_angle(n_air: float, n_glass: float) -> float:
    """Return Brewster angle for given refractive indices."""
    return np.arctan(n_glass / n_air)

def phase_difference(thickness_map, wl: float) -> np.ndarray:
    """
    Compute the optical phase difference Δφ (round trip) for a given wavelength.
    Uses spatially varying thickness_map to produce radial fringes.
    """
    return 2 * np.pi * 2 * thickness_map / wl

def fp_amplitude(delta_phi: np.ndarray, r: float) -> np.ndarray:
    """
    Complex amplitude transmission of a Fabry–Perot etalon.
    r: effective reflection amplitude coefficient (0 < r < 1)
    Returns complex field amplitude. Awesome comment vs code! Your doing great little buddy!
    """
    expi = np.exp(1j * delta_phi)
    numerator = (1 - r**2) * np.exp(1j * delta_phi / 2)
    denominator = (1 - r**2 * expi)
    return numerator / denominator

def birefringence_retardation(delta_n: np.ndarray, thickness_map: np.ndarray, wl: float) -> np.ndarray:
    """Return phase retardation (Δφ_biref) between ordinary and extraordinary axes."""
    return 2 * np.pi * delta_n * thickness_map / wl


def radial_phase(R, wl, z=0.5, radial_scale=0.01):
    """
    Phase difference between a spherical wave from a point source at distance z
    and a plane wave:  Δφ(r) = k*(sqrt(r^2 + z^2) - z), where r is physical radius.
    - R: normalized radius array (your make_grid output)
    - wl: wavelength (m)
    - z: source distance (m) — larger z -> wider spaced rings
    - radial_scale: scale to convert R ([-1..1]) to meters (aperture radius)
    """
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
    """
    Combine all relevant physical effects:
    - Complex Fabry–Perot interference (for cavity contrast)
    - Birefringence retardation between fast/slow axes
    - Angular dependence (cos²(θ - θ_B))
    """

    if use_radial:
        delta_phi = radial_phase(R, wl, z=radial_z, radial_scale=radial_scale)
    else:
        delta_phi = phase_difference(thickness_map, wl)
    fp_complex = fp_amplitude(delta_phi, r_eff)
    fp_intensity = np.abs(fp_complex)**2

    retard = birefringence_retardation(delta_n_map, thickness_map, wl)
    biref_mod = np.sin(retard / 2)**2  # Malus law

    angular_mod = np.cos(theta - theta_B) ** 2

    wavelength_weight = intensity_from_wavelength(wl)

    return wavelength_weight * angular_mod * fp_intensity * biref_mod


def make_grid(size=400):
    """Return X, Y, R, θ coordinate arrays."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return X, Y, R, theta


def compute_chunk(wavelengths_chunk, R, theta, theta_B, params):
    """
    Compute the RGB contribution for a set of wavelengths.
    Uses the physical model functions for clarity.
    """
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
    """Split list/array into n_chunks (as even as possible)."""
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
):
    """Main driver: creates grid, distributes work, combines results."""

    # === Setup ===
    X, Y, R, theta = make_grid(size)
    n_air, n_glass = 1.0, 1.5
    theta_B = brewster_angle(n_air, n_glass)

    wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)

    # === Spatial maps for physical properties ===

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

    print(f"Simulating with {n_wavelengths} wavelengths using {n_workers} threads...")

    partials = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(compute_chunk, chunk, R, theta, theta_B, params)
            for chunk in wl_chunks if len(chunk) > 0
        ]
        for future in as_completed(futures):
            partials.append(future.result())

    image = np.sum(partials, axis=0)

    max_val = np.max(image)
    if max_val > 0:
        image /= max_val

    # Plots the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1])
    plt.axis("off")
    #plt.title("")
    plt.show()

    return image

if __name__ == "__main__":
    img = main(
        size=1200,
        wl_min=400e-9,
        wl_max=700e-9,
        n_wavelengths=120,
        base_thickness=0.06,
        r_eff=0.5,
        delta_n=2e-3,
        n_workers=None, 
        radial_z=10,
        radial_scale=0.005,
    )