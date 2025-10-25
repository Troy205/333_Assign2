import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math

def get_component(image, colour: int, x0: int, y0: int, x1: int, y1: int):
    gradient = (y1-y0)/(x1-x0)
    X = list(map(math.floor, np.linspace(x0, x1, 500)))
    #print(X)
    Y = list(map(lambda x: math.floor(gradient * x + y0), X))
    new_image = image.copy()
    out = np.zeros((len(X)))
    new_image[Y, X] = [1, 0, 0]

    out = image[Y, X, colour]
    
    return (new_image, out)

#center at 1125, 1404
#yint at 0, 1578
polarised = img.imread("polar.jpeg")
normal = img.imread("normal.jpeg")

new_polar, red_line = get_component(polarised, 0, 0, 1578, 1125, 1404)
_, green_line = get_component(polarised, 1, 0, 1578, 1125, 1404)
_, blue_line = get_component(polarised, 2, 0, 1578, 1125, 1404)

#show image with line to show slice
plt.imshow(new_polar)

#if you want to plot the components
# plt.plot(red_line)
# plt.plot(green_line)
# plt.plot(blue_line)
plt.show()
