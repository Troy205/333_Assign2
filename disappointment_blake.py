import numpy as np
import math
import matplotlib.pyplot as plt

"""
I recommend collapsing the following functions.  I added a simple function
to calculate intensity distribution across wavelengths, and I added this term
to the intensity calculation. I also stole some code from stackexchange that
calculates the RGB fractions of each wavelength to apply colours to the image.

The result is the right colour! The rings are a bit more... suspicious. It's
indistinguishable for low finesse, but at the value I've set here they're 
somewhat visible.
"""

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

plt.show() # type: ignore
size = 800
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

n_air = 1.0
n_glass = 1.5
theta_B = np.arctan(n_glass/n_air) # Brewster angle calculation for light directly overhead

wavelengths = np.linspace(300e-9, 800-9, 50)
thickness_factor = 20000
F = 50000  # Fabry–Perot finesse

image = np.zeros((size, size, 3))

# Calculate the phase difference from whatever the hell goes on in glass (delta_phi)
# Then something to do with FP
# Finally we find the shift

for wl in wavelengths:
    delta_phi = 2 * np.pi * R * thickness_factor / wl

    fp_term = 1 / (1 + F * (np.sin(delta_phi / 2) ** 2))

    angular_mod = np.cos(theta - theta_B) ** 2
    
    #FP intensity scaled with spectral radiance from the Sun
    intensity = angular_mod * fp_term * intensity_from_wavelength(wl) 
    r, g, b = wavelength_to_rgb(wl * 1e9) # Get RGB fractions
    #Manipulate RGB components accordingly
    image[..., 0] += intensity * r
    image[..., 1] += intensity * g
    image[..., 2] += intensity * b

image /= image.max()

plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1]) # type: ignore
plt.axis("off") # type: ignore
plt.show() # type: ignore
