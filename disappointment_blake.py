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
size = 100
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

n_air = 1.0
n_glass = 1.5
theta_B = np.arctan(n_glass/n_air) # Brewster angle calculation for light directly overhead

wavelengths = np.linspace(300e-9, 800e-9, 50)
thickness_factor = 1e9
F = 0.65  # Fabry–Perot finesse

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
    for y in range(size):
        for x in range(size):
            #Manipulate RGB components accordingly
            image[y][x][0] += intensity[y][x] * r
            image[y][x][1] += intensity[y][x] * g
            image[y][x][2] += intensity[y][x] * b
maxi = 0
for y in range(size):
    for x in range(size):
        #Create rings
        #image[y][x][1] *= max(0, math.sin(1/3*math.sqrt((x-size/2)**2 + (y-size/2)**2)))
        
        #Log the intensities
        #for c in range(3):
        #    image[y][x][c] = max(0, math.log(max(0.1, image[y][x][c])))
        
        #Test the wavelength range you're scanning over
        # r, g, b = wavelength_to_rgb(wavelengths[(50*x)//size])
        # image[y][x][0] = r
        # image[y][x][1] = 0
        # image[y][x][2] = b
        
        #Divide each colour in each pixel by the maximum intensity pixel value
        maxi = max(maxi, math.sqrt(image[y][x][0]**2 + image[y][x][1]**2 + image[y][x][2]**2))
for y in range(size):
    for x in range(size):
        for c in range(3):
            image[y][x][c] /= maxi

plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1]) # type: ignore
plt.axis("off") # type: ignore
plt.show() # type: ignore

print(image[0, :, 0])
plt.plot(intensity[0, :]) #type: ignore
plt.show() #type: ignore