"""Scalar direct-light GGX/Smith/Schlick in linear RGB, evaluated on Warp.

Roughness is perceptual (alpha = roughness squared). Dielectric IOR is fixed
at 1.5 (F0=.04). No transmission, clearcoat, multiscattering or environment IBL.
Reference equations: https://google.github.io/filament/Filament.html
"""
import warp as wp


@wp.func
def direct_brdf(base: wp.vec3, roughness: float, metallic: float,
                normal: wp.vec3, view: wp.vec3, light: wp.vec3):
    nv = wp.max(wp.dot(normal, view), 0.0)
    nl = wp.max(wp.dot(normal, light), 0.0)
    if nv <= 0.0 or nl <= 0.0:
        return wp.vec3(0.0), wp.vec3(0.0)
    half = wp.normalize(view + light)
    nh = wp.clamp(wp.dot(normal, half), 0.0, 1.0)
    vh = wp.clamp(wp.dot(view, half), 0.0, 1.0)
    r = wp.clamp(roughness, 0.045, 1.0)
    metal = wp.clamp(metallic, 0.0, 1.0)
    a2 = r*r*r*r
    denominator = nh*nh*(a2-1.0)+1.0
    distribution = a2/(wp.pi*denominator*denominator)
    visibility = 0.5/(nl*wp.sqrt(nv*nv*(1.0-a2)+a2) + nv*wp.sqrt(nl*nl*(1.0-a2)+a2))
    f0 = wp.vec3(.04)*(1.0-metal)+base*metal
    fresnel = f0+(wp.vec3(1.0)-f0)*wp.pow(1.0-vh, 5.0)
    diffuse = wp.cw_mul(wp.vec3(1.0)-fresnel, base)*((1.0-metal)*nl/wp.pi)
    specular = fresnel*(distribution*visibility*nl)
    return diffuse, specular
