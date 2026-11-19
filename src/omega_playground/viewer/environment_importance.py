"""Energy-distributed environment samples and balance-heuristic MIS.

Reference: pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency
Preparation is once per environment, never per frame. PDF is per steradian.
"""
import numpy as np
import warp as wp
from .environment_radiance import sample_radiance
from .pbr import direct_brdf


@wp.struct
class EnvironmentDistribution:
    pdf: wp.array2d[float]
    samples: wp.array[wp.vec4]


def prepare_distribution(rgb, count, device):
    rgb = np.asarray(rgb, dtype=np.float64)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or not np.isfinite(rgb).all() or (rgb < 0).any() or count < 1):
        raise ValueError('Expected finite HDR RGB and positive sample count')
    h, w = rgb.shape[:2]
    edges = np.cos(np.arange(h+1)*np.pi/h)
    area = np.broadcast_to((edges[:-1]-edges[1:])[:, None]*2*np.pi/w, (h, w))
    # Include neighboring texels because bilinear radiance spreads bright texels
    # over their neighbors. The 1% uniform sphere component preserves support.
    luminance = rgb@np.array([.2126, .7152, .0722])
    horizontal = (.25*np.roll(luminance, 1, axis=1)+.5*luminance+.25*np.roll(luminance, -1, axis=1))
    padded = np.pad(horizontal, ((1, 1), (0, 0)), mode='edge')
    filtered = .25*padded[:-2]+.5*padded[1:-1]+.25*padded[2:]
    mass = filtered*area
    mass = mass/mass.sum() if mass.sum() > 0 else area/(4*np.pi)
    mass = .99*mass+.01*area/(4*np.pi)
    mass /= mass.sum()
    pdf = mass/area
    cdf = np.cumsum(mass.ravel())
    cdf[-1] = 1.
    indices = np.searchsorted(cdf, (np.arange(count)+.5)/count)
    rows, columns = indices//w, indices%w
    def radical(i, base):
        value, factor = 0., 1./base
        while i:
            value += (i%base)*factor
            i //= base
            factor /= base
        return value
    # Uniform longitude and cosine(theta) within selected spherical texel.
    u = np.array([radical(i+1, 2) for i in range(count)])
    v = np.array([radical(i+1, 3) for i in range(count)])
    longitude = np.pi-(columns+u)*2*np.pi/w
    y = edges[rows]*(1-v)+edges[rows+1]*v
    radius = np.sqrt(np.maximum(0., 1-y*y))
    directions = np.c_[radius*np.sin(longitude), y, radius*np.cos(longitude), pdf[rows, columns]]
    result = EnvironmentDistribution()
    result.pdf = wp.array(pdf.astype(np.float32), dtype=float, device=device)
    result.samples = wp.array(directions.astype(np.float32), dtype=wp.vec4, device=device)
    return result


@wp.func
def environment_pdf(distribution: EnvironmentDistribution, direction: wp.vec3):
    d = wp.normalize(direction)
    h = distribution.pdf.shape[0]
    w = distribution.pdf.shape[1]
    x = int(wp.floor((wp.pi-wp.atan2(d[0], d[2]))/(2.0*wp.pi)*float(w)))%w
    y = wp.min(int(wp.acos(wp.clamp(d[1], -1., 1.))/wp.pi*float(h)), h-1)
    return distribution.pdf[y, x]


@wp.func
def ggx_pdf(normal: wp.vec3, view: wp.vec3, light: wp.vec3, roughness: float):
    result = float(0.)
    half = wp.normalize(view+light)
    nh = wp.dot(normal, half)
    vh = wp.dot(view, half)
    if nh > 0. and vh > 0.:
        r = wp.clamp(roughness, .045, 1.)
        a2 = r*r*r*r
        denominator = nh*nh*(a2-1.)+1.
        result = a2*nh/(wp.pi*denominator*denominator*4.*vh)
    return result


@wp.func
def mis_environment(pixels: wp.array2d[wp.vec3], distribution: EnvironmentDistribution,
                    rotation: wp.mat33, gain: wp.vec3, base: wp.vec3,
                    roughness: float, metallic: float, normal: wp.vec3, view: wp.vec3, count: int):
    result = wp.vec3(0.)
    if wp.dot(normal, view) > 0.:
        up = wp.vec3(0., 0., 1.)
        if wp.abs(normal[2]) > .999:
            up = wp.vec3(1., 0., 0.)
        tangent = wp.normalize(wp.cross(up, normal))
        bitangent = wp.cross(normal, tangent)
        r = wp.clamp(roughness, .045, 1.)
        a2 = r*r*r*r
        for i in range(count):
            bits = int(i)
            fraction = float(0.)
            power = float(.5)
            for j in range(16):
                fraction += float(bits%2)*power
                bits = bits/2
                power *= .5
            u = (float(i)+.5)/float(count)
            ct = wp.sqrt((1.-u)/(1.+(a2-1.)*u))
            st = wp.sqrt(wp.max(0., 1.-ct*ct))
            phi = 2.*wp.pi*fraction
            half = tangent*(st*wp.cos(phi))+bitangent*(st*wp.sin(phi))+normal*ct
            light = 2.*wp.dot(view, half)*half-view
            if wp.dot(view, half) > 0. and wp.dot(normal, light) > 0.:
                pb = ggx_pdf(normal, view, light, roughness)
                pe = environment_pdf(distribution, rotation*light)
                diffuse, specular = direct_brdf(base, roughness, metallic, normal, view, light)
                result += wp.cw_mul(wp.cw_mul(sample_radiance(pixels, rotation*light), gain), specular)/(pb+pe)
            sample = distribution.samples[i]
            local = wp.vec3(sample[0], sample[1], sample[2])
            light = wp.transpose(rotation)*local
            if wp.dot(normal, light) > 0.:
                pb = ggx_pdf(normal, view, light, roughness)
                diffuse, specular = direct_brdf(base, roughness, metallic, normal, view, light)
                result += wp.cw_mul(wp.cw_mul(sample_radiance(pixels, local), gain), specular)/(pb+sample[3])
        result /= float(count)
    return result
