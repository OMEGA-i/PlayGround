"""sRGB transfer functions; no tone mapping or exposure gain."""
import warp as wp


@wp.func
def linear_to_srgb(x: float):
    value = wp.max(x, 0.0)
    if value <= 0.0031308:
        return 12.92*value
    return 1.055*wp.pow(value, 1.0/2.4)-0.055


@wp.func
def srgb_to_linear(x: float):
    value = wp.max(x, 0.0)
    if value <= 0.04045:
        return value/12.92
    return wp.pow((value+0.055)/1.055, 2.4)


@wp.func
def display_srgb(rgb: wp.vec3):
    return wp.vec3(linear_to_srgb(rgb[0]), linear_to_srgb(rgb[1]), linear_to_srgb(rgb[2]))
