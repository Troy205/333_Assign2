import numpy as np
import matplotlib.pyplot as plt

size = 800
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

n_air = 1.0
n_glass = 1.5
theta_B = np.arctan(n_glass/n_air) # Brewster angle calculation for light directly overhead

wavelengths = {"R": 650, "G": 530, "B": 450}
thickness_factor = 2000
F = 5  # Fabry–Perot finesse

image = np.zeros((size, size, 3))

# Calculate the phase difference from whatever the hell goes on in glass (delta_phi)
# Then something to do with FP
# Finally we find the shift

for i, (color, wl) in enumerate(wavelengths.items()):
    delta_phi = 2 * np.pi * R * thickness_factor / wl

    fp_term = 1 / (1 + F * (np.sin(delta_phi / 2) ** 2))

    angular_mod = np.cos(theta - theta_B) ** 2

    intensity = angular_mod * fp_term
    image[..., i] = intensity

image /= image.max()

plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1])
plt.axis("off")
plt.show()