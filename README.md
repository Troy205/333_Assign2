
# Physics 333 Assignment 2

Figure_1.png is the raw render of my model
polar_components.png is the rbg components along the slice shown in polar_sliced.png, from out toward the center
similarly, render_components.png is the rgb components along the slice shown in render_sliced.png from out toward the center

code:
original_method.py
ill admit we kind of have some arbitrary physical params i didnt even try changing(there are so many), but they are, lets say, suitable
the code scans over a linear distribution of wavelengths over the visible spectrum, and for each one:
calculates and angular term which basically dims the light the closer it is to brewster angle
the base intensity is the light from the sun as a black body at 5777K, and we split the wavelength into rgb colour channels to render an image
however, we also want to take the phase shift into account
this is proportinal to 2 * pi * thickness * n1 * math.cos(refraction_angle)/wavelength of the ray into the material
to find the angle, uhhh
basically find the angle the ray takes from the glass to the observer using atan(sqrt(x^2+y^2)/distance to glass)
then use snells law to find the angle of the ray inside the glass
then apply interference using cos^2(phase/2) (im not sure about this step)
then normalise the image by dividing the intensity of each pixel by the maximum intensity pixel's intensity

explanation for discrepancy:
lack of tuning physical parameters: changing thickness for example a tiny bit results in different favoured hues


More formal write up: (this might turn out bad cuz honestly im a bit cooked rn)

Method:
This approach uses a simple phase-shift method to derive the interference. We approximate sunlight to be blackbody radiation, modelled at 5777K, and uniformly sample across wavelengths, giving a base intensity for each wavelength $\lambda$. We then provide an angular modulation term, to account for rays entering close to Brewster's angle. The phase shift is due to birefringence in the glass, so it will look something like $\frac{2\pi n d \cos(\theta)}{\lambda}$, where n is the refractive index of the glass, d is the thickness of the glass, and $\theta$ is the angle to the normal the ray inside the glass travels. d is a free variable(as long as it remains a reasonable number), and it controls what frequencies are favoured in the fringes of the resultant interference pattern. $\theta$ we can calculate, as we can find the angle of incidence to the glass at that pixel in the image using trigonometry, and then use Snell's law to find the angle of the ray inside the glass. We then apply a similar interference step as the angular modulation, with a $\cos^2$ term. Finally, after adding up the contributions to the image from all the wavelengths, we normalise the image by dividing each pixel's colour channel by the maximum intensity pixel's intensity, to preserve relative pixel colours.

Limitations:
This model is very sensitive to free parameter changes, mainly the thickness of the glass, and the number of wavelengths sampled. The former is expected, as that would determine what wavelengths interfere constructively or destructively in the interference pattern. The latter is more inexplicable; for development purposes, we sampled 200 wavelengths from 300-800nm, but for large sample sizes, e.g. 500, this model produced essentially white light. I expect this is a symptom of either the chosen normalisation method being unsuitable, or the result of floating point errors due to relatively large intensities compared to individual wavelength contributions.

Results:
There are some qualitative similarities: they both contain a relatively bright centre, followed by a yellowish fringe, followed by sequential, close-to equidistant fringes. They both contain regions where the pattern vanishes. However, there are also some qualitative dissimilarities: the model clearly predicts the wrong frequencies. This is likely because of its sensitivity to the free parameters, and without some automation to tune them, picking the right ones is very difficult. The camera image also has much more saturated colours in general, suggesting the normalisation technique is malformed. The centre of each figure also differs, the camera image having a more circular feature than the render. At this stage, it is unclear whether this is indicative of another, yet-unmodeled phenomena, or if imposed on a sky-blue background, a similar effect could be achieved by adjusting the free physical parameters.

These plots depict the colour channels traced along their respective images' lines, from the outside of the image toward the center of the phenomena, about 5 fringes total. In the plot from the photograph, there is an anomalous spike in all three colour channels due to the reflection of a light in the image, so we will discount that from our discussion. In general, across both plots, despite containing different fractions of the intensity, there is a red peak where there is a green trough, resulting in their respective magenta and green/cyan fringes. In both plots, all three channels oscillate with different frequencies, as expected. However, the plot of the photograph also clearly shows a decay of intensity with distance from the centre, which our model failed to incorporate. This decay would be due to conservation of energy.