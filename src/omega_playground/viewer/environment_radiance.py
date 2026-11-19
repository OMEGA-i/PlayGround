"""Linear HDR latlong sampling for environment integration on Warp.

Directions are in the USD DomeLight local frame (+Y north, +Z longitude zero).
No exposure, normalization, tone mapping or sRGB transfer is applied here.
"""
import warp as wp


@wp.func
def sample_radiance(pixels: wp.array2d[wp.vec3], direction: wp.vec3):
    d = wp.normalize(direction)
    height = pixels.shape[0]
    width = pixels.shape[1]
    u = (wp.pi-wp.atan2(d[0], d[2]))/(2.0*wp.pi)*float(width)-0.5
    v = wp.clamp(wp.acos(wp.clamp(d[1], -1.0, 1.0))/wp.pi*float(height)-0.5,
                 0.0, float(height-1))
    x = int(wp.floor(u))
    y = int(wp.floor(v))
    a = u-float(x)
    b = v-float(y)
    x0 = (x+width)%width
    x1 = (x0+1)%width
    y1 = wp.min(y+1, height-1)
    return ((1.0-a)*(1.0-b)*pixels[y, x0]+a*(1.0-b)*pixels[y, x1]
            +(1.0-a)*b*pixels[y1, x0]+a*b*pixels[y1, x1])


def upload_radiance(rgb, device):
    """Upload once; caller owns the resulting array for the renderer lifetime."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.float32)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or not np.isfinite(rgb).all() or (rgb < 0).any()):
        raise ValueError('Expected nonempty finite nonnegative linear HDR RGB')
    return wp.array(rgb, dtype=wp.vec3, device=device)


@wp.func
def specular_environment(pixels: wp.array2d[wp.vec3], world_to_dome: wp.mat33,
                         gain: wp.vec3, base: wp.vec3, roughness: float, metallic: float,
                         normal: wp.vec3, view: wp.vec3, count: int):
    # Deterministic low-discrepancy GGX NDF importance sampling. Unoccluded,
    # single scattering; this is an integration baseline, not a prefiltered IBL.
    result = wp.vec3(0.0)
    nv = wp.max(wp.dot(normal, view), 0.0)
    if nv > 0.0:
        up = wp.vec3(0.0, 0.0, 1.0)
        if wp.abs(normal[2]) > 0.999:
            up = wp.vec3(1.0, 0.0, 0.0)
        tangent = wp.normalize(wp.cross(up, normal))
        bitangent = wp.cross(normal, tangent)
        r = wp.clamp(roughness, .045, 1.0)
        a2 = r*r*r*r
        metal = wp.clamp(metallic, 0.0, 1.0)
        f0 = wp.vec3(.04)*(1.0-metal)+base*metal
        for i in range(count):
            bits = int(i)
            fraction = float(0.0)
            power = float(.5)
            for j in range(16):
                fraction += float(bits%2)*power
                bits = bits/2
                power *= .5
            u = (float(i)+.5)/float(count)
            phi = 2.0*wp.pi*fraction
            ct = wp.sqrt((1.0-u)/(1.0+(a2-1.0)*u))
            st = wp.sqrt(wp.max(0.0, 1.0-ct*ct))
            half = tangent*(st*wp.cos(phi))+bitangent*(st*wp.sin(phi))+normal*ct
            vh = wp.dot(view, half)
            light = 2.0*vh*half-view
            nl = wp.dot(normal, light)
            if vh > 0.0 and nl > 0.0:
                visibility = .5/(nl*wp.sqrt(nv*nv*(1.0-a2)+a2)
                                  +nv*wp.sqrt(nl*nl*(1.0-a2)+a2))
                fresnel = f0+(wp.vec3(1.0)-f0)*wp.pow(1.0-vh, 5.0)
                # (D*V*F*NoL) / (D*NoH/(4*VoH)); D cancels.
                weight = visibility*nl*4.0*vh/ct
                radiance = wp.cw_mul(sample_radiance(pixels, world_to_dome*light), gain)
                result += wp.cw_mul(radiance, fresnel)*weight
        result /= float(count)
    return result
