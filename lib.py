import math
def intensity_from_wavelength(wl: float) -> float:
    """_summary_
    Treating the Sun as an ideal blackbody at 5777K.

    Args:
        wl (float): Input wavelength

    Returns:
        float: Relative intensity of this wavelength as emitted by an ideal blackbody of 5777K.
    """
    # Ok that docstring was entirely unnecessary but whatever
    t = 5777
    c = 3e8
    h = 6.62607015e-34
    k = 1.380649e-23
    return h*c**2/wl**5 / (math.exp(h*c/wl/k/t) - 1)

# The following code is taken from https://stackoverflow.com/a/34581745
def wavelength_to_rgb(wavelength: float) -> tuple[float, float, float]:
    xyz = cie1931_wavelength_to_xyz_fit(wavelength)
    rgb = srgb_xyz_to_rgb(xyz)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (r, g, b)
def srgb_xyz_to_rgb(xyz: list[float]) -> list[float]:
    x, y, z = xyz

    rl =  3.2406255 * x + -1.537208  * y + -0.4986286 * z
    gl = -0.9689307 * x +  1.8757561 * y +  0.0415175 * z
    bl =  0.0557101 * x + -0.2040211 * y +  1.0569959 * z

    return [
        srgb_postprocess(rl),
        srgb_postprocess(gl),
        srgb_postprocess(bl)
    ]
def srgb_postprocess(c: float) -> float:
    # clip to [0,1]
    c = 1 if c > 1 else (0 if c < 0 else c)

    # apply transfer function
    if c <= 0.0031308:
        return c * 12.92
    else:
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055
def cie1931_wavelength_to_xyz_fit(wave: float) -> list[float]:
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