# lib.py
import math
import numpy as np

# ---------------------------
# Spectral radiance (Sun)
# ---------------------------
def intensity_from_wavelength(wl: float) -> float:
    """
    Spectral radiance (relative) of a blackbody at T=5777 K.
    wl: wavelength in meters
    returns: spectral radiance (linear units, proportional to W/sr/m^3 but we treat relative)
    """
    # physical constants
    t = 5777.0
    c = 299792458.0
    h = 6.62607015e-34
    k = 1.380649e-23
    # Planck's law (spectral radiance per unit wavelength)
    # B_lambda = (2*h*c^2 / wl^5) * 1/(exp(h*c/(wl*k*T)) - 1)
    # drop the leading 2 as we only need relative scale
    val = (h * c**2) / (wl**5) / (math.exp(h * c / (wl * k * t)) - 1.0)
    return val


# ---------------------------
# CIE 1931 fit (returns X,Y,Z relative)
# ---------------------------
def cie1931_wavelength_to_xyz_fit(wave: float) -> list[float]:
    """
    Analytic fit (approx) to CIE 1931 color matching functions.
    wave: wavelength in nm (380-780 recommended)
    returns [X, Y, Z] relative (not scaled to any absolute illuminant)
    """
    # X
    t1 = (wave - 442.0) * (0.0624 if wave < 442.0 else 0.0374)
    t2 = (wave - 599.8) * (0.0264 if wave < 599.8 else 0.0323)
    t3 = (wave - 501.1) * (0.0490 if wave < 501.1 else 0.0382)
    x = (0.362 * math.exp(-0.5 * t1 * t1)
       + 1.056 * math.exp(-0.5 * t2 * t2)
       - 0.065 * math.exp(-0.5 * t3 * t3))

    # Y
    t1 = (wave - 568.8) * (0.0213 if wave < 568.8 else 0.0247)
    t2 = (wave - 530.9) * (0.0613 if wave < 530.9 else 0.0322)
    y = (0.821 * math.exp(-0.5 * t1 * t1)
       + 0.286 * math.exp(-0.5 * t2 * t2))

    # Z
    t1 = (wave - 437.0) * (0.0845 if wave < 437.0 else 0.0278)
    t2 = (wave - 459.0) * (0.0385 if wave < 459.0 else 0.0725)
    z = (1.217 * math.exp(-0.5 * t1 * t1)
       + 0.681 * math.exp(-0.5 * t2 * t2))

    return [x, y, z]


# ---------------------------
# XYZ <-> linear sRGB conversion
# ---------------------------
# Matrix from CIE XYZ (D65) to linear sRGB
# sRGB linear = M * [X, Y, Z]
_M_XYZ_TO_sRGB = np.array([
    [ 3.2406255, -1.537208 , -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959],
], dtype=float)

def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to linear sRGB (may be out of gamut or negative)."""
    # xyz shape (...,3)
    # multiply by matrix (works for vector or single)
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        lr = _M_XYZ_TO_sRGB.dot(xyz)
        return lr
    else:
        # shape (...,3) -> (...,3)
        return xyz.dot(_M_XYZ_TO_sRGB.T)


def linear_srgb_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """
    Apply the sRGB transfer function (gamma) and clip to [0,1].
    Accepts array or single triple. Returns clipped sRGB.
    """
    def _transfer(c):
        # assume c is scalar or numpy array
        c = np.array(c, dtype=float)
        # clip negative before applying transfer to avoid NaNs
        c = np.clip(c, 0.0, None)
        mask = c <= 0.0031308
        out = np.empty_like(c)
        out[mask] = c[mask] * 12.92
        out[~mask] = 1.055 * (c[~mask] ** (1.0 / 2.4)) - 0.055
        return np.clip(out, 0.0, 1.0)

    return _transfer(linear_rgb)


# ---------------------------
# Per-wavelength linear RGB (monochromatic) — returns linear RGB
# ---------------------------
def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """
    Convert a single wavelength (nm) to linear RGB triple (not gamma corrected).
    This uses the CIE fit to get XYZ then converts to linear sRGB.
    Returns (R_lin, G_lin, B_lin).
    """
    xyz = cie1931_wavelength_to_xyz_fit(wavelength_nm)
    linear_rgb = xyz_to_linear_srgb(np.array(xyz))
    # Note: values may be negative (out of sRGB gamut) for some wavelengths; we keep them
    # The caller should integrate linear RGB and only apply gamma at the end.
    return (float(linear_rgb[0]), float(linear_rgb[1]), float(linear_rgb[2]))


# ---------------------------
# Spectral integration: wavelengths (m) + spectral radiance -> display sRGB
# ---------------------------
def spectrum_to_srgb(wavelengths_m: np.ndarray, spectral_radiance: np.ndarray, delta_lambda=None) -> np.ndarray:
    """
    Convert a sampled spectrum to display-ready sRGB.
    wavelengths_m : 1D array (meters)
    spectral_radiance : same-length array (linear units)
    delta_lambda : spacing in meters (if None, inferred from wavelengths)
    Returns sRGB triple in [0,1]
    """
    wavelengths_nm = np.asarray(wavelengths_m, dtype=float) * 1e9
    spec = np.asarray(spectral_radiance, dtype=float)
    if delta_lambda is None:
        # approximate uniform spacing
        diffs = np.diff(wavelengths_nm)
        if diffs.size == 0:
            dl = 1.0
        else:
            dl = np.mean(diffs)
    else:
        dl = float(delta_lambda * 1e9)  # convert meters->nm if provided in m

    # accumulate XYZ by integrating radiance * CMF * dl
    X = 0.0
    Y = 0.0
    Z = 0.0
    for wl_nm, L in zip(wavelengths_nm, spec):
        x, y, z = cie1931_wavelength_to_xyz_fit(wl_nm)
        X += L * x * dl
        Y += L * y * dl
        Z += L * z * dl

    # Convert to linear RGB
    XYZ = np.array([X, Y, Z], dtype=float)
    # Small normalization to avoid huge values; keep relative color
    if XYZ.sum() <= 0:
        linear_rgb = np.zeros(3)
    else:
        linear_rgb = xyz_to_linear_srgb(XYZ)

    # Now apply simple autoscaling so that max channel ≤ 1 before gamma (preserving color)
    maxc = np.max(linear_rgb)
    if maxc > 0:
        linear_rgb = linear_rgb / maxc

    # Apply sRGB gamma transfer and clip
    srgb = linear_srgb_to_srgb(linear_rgb)
    return np.clip(srgb, 0.0, 1.0)


# ---------------------------
# Convenience: convert an image of linear RGB to sRGB for display
# ---------------------------
def linear_image_to_srgb_image(img_linear: np.ndarray, clip_percentile=99.2, saturation_boost=1.0):
    """
    img_linear : HxWx3 linear RGB image (float)
    Returns HxWx3 sRGB image in [0,1].
    - Performs percentile exposure scaling for robustness
    - Optional simple saturation boost (implemented in HSV if needed externally)
    """
    import numpy as _np
    from matplotlib import colors as _colors

    img = _np.array(img_linear, dtype=float)
    # luminance (Rec.709)
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    scale = _np.percentile(lum, clip_percentile)
    if scale <= 0:
        scale = lum.max() if lum.max() > 0 else 1.0
    img = img / scale

    # clamp negative (out-of-gamut) values to zero before gamut mapping
    img = _np.clip(img, 0.0, None)

    # saturation boost if requested (done in HSV)
    if saturation_boost != 1.0:
        hsv = _colors.rgb_to_hsv(_np.clip(img, 0.0, 1.0))
        hsv[..., 1] = _np.clip(hsv[..., 1] * saturation_boost, 0.0, 1.0)
        img = _colors.hsv_to_rgb(hsv)

    # apply sRGB gamma
    img = linear_srgb_to_srgb(img)
    img = _np.clip(img, 0.0, 1.0)
    return img