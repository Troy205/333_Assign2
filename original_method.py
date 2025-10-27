import numpy as np
import matplotlib.pyplot as plt
from lib import *
"""
                                                I added a simple function
to calculate intensity distribution across wavelengths, and I added this term
to the intensity calculation. I also stole some code from stackexchange that
calculates the RGB fractions of each wavelength to apply colours to the image.

The result is the right colour! The rings are a bit more... suspicious. It's
indistinguishable for low finesse, but at the value I've set here they're 
somewhat visible.
"""
size = 500
zoom = 0.8
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

n_air = 1.0
n_glass = 1.5
theta_B = np.arctan(n_glass/n_air) # Brewster angle calculation for light directly overhead

wavelengths = np.linspace(300e-9, 800e-9, 200)
thickness_factor = 1e9
F = 0.65  # Fabry–Perot finesse

image = np.zeros((size, size, 3))

# Calculate the phase difference from whatever the hell goes on in glass (delta_phi)
# Then something to do with FP
# Finally we find the shift
distance_from_glass = 1
for wl in wavelengths:
    delta_phi = 2 * np.pi * R * thickness_factor / wl

    fp_term = 1 / (1 + F * (np.sin(delta_phi / 2) ** 2))

    angular_mod = np.cos(theta - theta_B) ** 2
    
    #FP intensity scaled with spectral radiance from the Sun
    intensity = angular_mod *  intensity_from_wavelength(wl) 
    r, g, b = wavelength_to_rgb(wl * 1e9) # Get RGB fractions
    for y in range(size):
        for x in range(size):
            incidence = math.atan(math.sqrt(((x-size/2)/(size/zoom))**2 + ((y-size/2)/(size/zoom))**2)/distance_from_glass)
            refraction = math.asin(n_air/n_glass * math.sin(incidence))
            phase = phase_shift(n_glass, 1e-4, refraction, wl)
            intensity[y][x] *= math.cos(phase/2)**2
            #Manipulate RGB components accordingly
            image[y][x][0] += intensity[y][x] * r
            image[y][x][1] += intensity[y][x] * g
            image[y][x][2] += intensity[y][x] * b
# print(image)
# sky_r, sky_g, sky_b = 180/255, 210/255, 246/255
# blue_amout = 1e14
# for y in range(size):
#     for x in range(size):
#         image[y][x][0] += sky_r * blue_amout
#         image[y][x][1] += sky_g * blue_amout
#         image[y][x][2] += sky_b * blue_amout
maxi = 0
for y in range(size):
    for x in range(size):
        maxi = max(maxi, math.sqrt(image[y][x][0]**2 + image[y][x][1]**2 + image[y][x][2]**2))
for y in range(size):
    for x in range(size):
        for c in range(3):
            image[y][x][c] /= maxi

plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1]) # type: ignore
plt.axis("off") # type: ignore
plt.show() # type: ignore