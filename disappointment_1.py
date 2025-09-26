import numpy as np
import matplotlib.pyplot as plt

# This code achieves nothing but producing an image. There is no real physics here.
# This file uses some chatGPT, do not include in this form in final project.

size = 800
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

wavelengths = {
    "R": 650,
    "G": 530,
    "B": 450
}

thickness_factor = 2000
phase = 2 * np.pi * R * thickness_factor

angular_modulation = np.sin(theta) ** 2

image = np.zeros((size, size, 3))

for i, (color, wl) in enumerate(wavelengths.items()):

    intensity = 0.5 * (1 + np.cos(phase / wl))
    intensity *= angular_modulation 
    image[..., i] = intensity

image = image / image.max()

plt.imshow(image, origin="lower", extent=[-1, 1, -1, 1])
plt.axis("off")
plt.show()
