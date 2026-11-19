"""Offline Radiance RGBE decode and solid-angle weighted latlong light bins.

USD DomeLight uses +Y at the north pole and +Z at longitude zero:
https://openusd.org/release/user_guides/schemas/usdLux/DomeLight.html
Only modern RGBE RLE with the standard -Y/+X storage order is accepted.
"""
import re
from pathlib import Path

import numpy as np


def read_rgbe(path):
    with Path(path).open('rb') as stream:
        if stream.readline().strip() not in (b'#?RADIANCE', b'#?RGBE'):
            raise ValueError('Not a Radiance HDR image')
        header = []
        for _ in range(100):
            line = stream.readline()
            if not line:
                raise ValueError('Truncated HDR header')
            if not line.strip():
                break
            header.append(line.strip())
        else:
            raise ValueError('HDR header too long')
        if b'FORMAT=32-bit_rle_rgbe' not in header:
            raise ValueError('Only RGBE Radiance images are supported')
        if any(x.startswith((b'EXPOSURE=', b'COLORCORR=')) for x in header):
            raise ValueError('HDR exposure/color correction headers need explicit conversion')
        shape = re.fullmatch(rb'-Y (\d+) \+X (\d+)\s*', stream.readline())
        if shape is None:
            raise ValueError('Only -Y/+X HDR orientation is supported')
        height, width = map(int, shape.groups())
        if not 8 <= width <= 32767 or not 1 <= height or height*width > 64_000_000:
            raise ValueError('Unsupported HDR dimensions')
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        for row in range(height):
            if stream.read(4) != bytes((2, 2, width >> 8, width & 255)):
                raise ValueError('Unsupported or truncated HDR scanline')
            for channel in range(4):
                offset = 0
                while offset < width:
                    marker = stream.read(1)
                    if not marker or marker == b'\0':
                        raise ValueError('Truncated or invalid HDR run')
                    count = marker[0]
                    repeat = count > 128
                    count = count-128 if repeat else count
                    data = stream.read(1 if repeat else count)
                    if offset+count > width or len(data) != (1 if repeat else count):
                        raise ValueError('HDR run exceeds scanline or is truncated')
                    pixels[row, offset:offset+count, channel] = np.frombuffer(data, dtype=np.uint8)
                    offset += count
    exponent = pixels[:, :, 3].astype(np.int32)
    result = np.ldexp(pixels[:, :, :3].astype(np.float64), exponent[:, :, None]-136)
    result[exponent == 0] = 0
    return result


def light_bins(rgb, rotation, rows=4, columns=4):
    """Return incoming world directions and RGB integral / (pi * mean luminance).

    Binning is a diffuse quadrature approximation, not environment reflections.
    All input texels contribute with their actual spherical solid angle.
    """
    rgb = np.asarray(rgb, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all() or (rgb < 0).any():
        raise ValueError('Expected finite nonnegative HDR RGB')
    if rotation.shape != (3, 3) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5):
        raise ValueError('Dome transform must be a rotation')
    height, width, _ = rgb.shape
    if height < rows or width < columns:
        raise ValueError('HDR image smaller than light grid')
    theta = (np.arange(height)+.5)*np.pi/height
    longitude = np.pi-(np.arange(width)+.5)*2*np.pi/width
    weights = (np.cos(np.arange(height)*np.pi/height)-
               np.cos((np.arange(height)+1)*np.pi/height))*2*np.pi/width
    mean_rgb = np.sum(rgb*weights[:, None, None], axis=(0, 1))/(4*np.pi)
    mean_luminance = float(mean_rgb@np.array([.2126, .7152, .0722]))
    bins = []
    for ys in np.array_split(np.arange(height), rows):
        for xs in np.array_split(np.arange(width), columns):
            incoming = np.stack(np.broadcast_arrays(
                np.sin(theta[ys, None])*np.sin(longitude[xs]),
                np.cos(theta[ys, None]),
                np.sin(theta[ys, None])*np.cos(longitude[xs])), axis=-1)
            area = weights[ys, None]
            values = rgb[ys[:, None], xs]*area[:, :, None]
            energy = values.sum(axis=(0, 1))
            brightness = values@np.array([.2126, .7152, .0722])
            vector = (incoming*brightness[:, :, None]).sum(axis=(0, 1))
            if np.linalg.norm(vector) < 1e-15 or mean_luminance < 1e-15:
                continue
            direction = rotation@(vector/np.linalg.norm(vector))
            bins.append({'incoming': direction.tolist(),
                         'diffuse': (energy/(np.pi*mean_luminance)).tolist()})
    return bins, {'mean_rgb': mean_rgb.tolist(), 'mean_luminance': mean_luminance,
                  'grid': [rows, columns], 'light_count': len(bins)}


def cubemap_linear(rgb, rotation, size=256):
    """Sample USD latlong into MJWarp/MuJoCo's six vertically stacked faces.

    Inverse of MJWarp sample_skybox: cube coordinates are (world.x, world.z,
    -world.y), faces +X,-X,+Y,-Y,+Z,-Z. Sampling is bilinear with U wrapping.
    """
    height, width, _ = rgb.shape
    s, t = np.meshgrid(2*(np.arange(size)+.5)/size-1, 2*(np.arange(size)+.5)/size-1)
    one = np.ones_like(s)
    world = np.stack([np.stack(x, axis=-1) for x in
                      [(one, s, -t), (-one, -s, -t), (s, -t, one),
                       (s, t, -one), (s, -one, -t), (-s, one, -t)]])
    local = world@rotation
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    u = (np.pi-np.arctan2(local[..., 0], local[..., 2]))/(2*np.pi)*width-.5
    v = np.clip((.5-np.arcsin(np.clip(local[..., 1], -1, 1))/np.pi)*height-.5, 0, height-1)
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    a, b = (u-x)[..., None], (v-y)[..., None]
    result = ((1-a)*(1-b)*rgb[y, x % width]+a*(1-b)*rgb[y, (x+1) % width]+
              (1-a)*b*rgb[np.minimum(y+1, height-1), x % width]+
              a*b*rgb[np.minimum(y+1, height-1), (x+1) % width])
    return result.reshape(6*size, size, 3)


def skybox_bytes(rgb, rotation, mean_luminance, color=(1, 1, 1), size=256):
    linear = cubemap_linear(rgb, rotation, size)*np.asarray(color)/max(mean_luminance, 1e-12)
    mapped = linear/(1+linear)  # Explicit display-only Reinhard tone mapping.
    srgb = np.where(mapped <= .0031308, 12.92*mapped, 1.055*mapped**(1/2.4)-.055)
    return (np.clip(srgb, 0, 1)*255).round().astype(np.uint8).tobytes()
