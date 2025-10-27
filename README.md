
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