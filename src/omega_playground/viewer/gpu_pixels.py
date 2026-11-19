"""Quantize rendered RGB on device before the encoder's host readback boundary."""
import warp as wp


@wp.kernel
def _quantize(source: wp.array3d(dtype=wp.vec3), target: wp.array4d(dtype=wp.uint8)):
    n, y, x, c = wp.tid()
    value = source[n, y, x][c]
    if wp.isnan(value):
        target[n, y, x, c] = wp.uint8(0)
    else:
        scaled = wp.clamp(value, 0.0, 1.0) * 255.0
        lower = wp.floor(scaled)
        integer = int(lower)
        fraction = scaled-lower
        # numpy.round uses ties-to-even; CUDA roundf instead rounds ties away.
        if fraction > 0.5 or (fraction == 0.5 and integer % 2 != 0):
            integer += 1
        target[n, y, x, c] = wp.uint8(integer)


@wp.kernel
def _resolve(source: wp.array3d(dtype=wp.vec3), target: wp.array3d(dtype=wp.vec3), scale: int):
    n, y, x = wp.tid()
    total = wp.vec3(0.0)
    for dy in range(scale):
        for dx in range(scale):
            color = source[n, y*scale+dy, x*scale+dx]
            for c in range(3):
                v = wp.clamp(color[c], 0.0, 1.0)
                if wp.isnan(v):
                    v = 0.0
                linear = v / 12.92
                if v > 0.04045:
                    linear = wp.pow((v+0.055)/1.055, 2.4)
                total[c] += linear
    for c in range(3):
        linear = total[c] / float(scale*scale)
        value = 12.92*linear
        if linear > 0.0031308:
            value = 1.055*wp.pow(linear, 1.0/2.4)-0.055
        total[c] = value
    target[n, y, x] = total


@wp.func
def _sample(image: wp.array3d(dtype=wp.vec3), n: int, x: float, y: float):
    x = wp.clamp(x, 0.0, float(image.shape[2]-1))
    y = wp.clamp(y, 0.0, float(image.shape[1]-1))
    ix, iy = int(wp.floor(x)), int(wp.floor(y))
    fx, fy = x-float(ix), y-float(iy)
    jx, jy = wp.min(ix+1, image.shape[2]-1), wp.min(iy+1, image.shape[1]-1)
    return wp.lerp(wp.lerp(image[n,iy,ix], image[n,iy,jx], fx),
                   wp.lerp(image[n,jy,ix], image[n,jy,jx], fx), fy)


@wp.kernel
def _edge_resolve(source: wp.array3d(dtype=wp.vec3), target: wp.array3d(dtype=wp.vec3)):
    n, y, x = wp.tid()
    xf, yf = float(x), float(y)
    luma = wp.vec3(0.299, 0.587, 0.114)
    center = source[n,y,x]
    nw = wp.dot(_sample(source,n,xf-1.0,yf-1.0),luma)
    ne = wp.dot(_sample(source,n,xf+1.0,yf-1.0),luma)
    sw = wp.dot(_sample(source,n,xf-1.0,yf+1.0),luma)
    se = wp.dot(_sample(source,n,xf+1.0,yf+1.0),luma)
    mid = wp.dot(center,luma)
    lo = wp.min(mid,wp.min(wp.min(nw,ne),wp.min(sw,se)))
    hi = wp.max(mid,wp.max(wp.max(nw,ne),wp.max(sw,se)))
    result = center
    if hi-lo > wp.max(0.0312,hi*0.125):
        dx, dy = -((nw+ne)-(sw+se)), (nw+sw)-(ne+se)
        reduce = wp.max((nw+ne+sw+se)*0.03125,0.0078125)
        inverse = 1.0/(wp.min(wp.abs(dx),wp.abs(dy))+reduce)
        dx, dy = wp.clamp(dx*inverse,-8.0,8.0), wp.clamp(dy*inverse,-8.0,8.0)
        a = 0.5*(_sample(source,n,xf-dx/6.0,yf-dy/6.0)+_sample(source,n,xf+dx/6.0,yf+dy/6.0))
        b = 0.5*a+0.25*(_sample(source,n,xf-dx*0.5,yf-dy*0.5)+_sample(source,n,xf+dx*0.5,yf+dy*0.5))
        value = wp.dot(b,luma)
        result = b
        if value < lo or value > hi:
            result = a
    target[n,y,x] = result


class Rgb8Readback:
    def __init__(self, shape, device, supersample=1, edge_antialias=False):
        if type(supersample) is not int or supersample not in (1, 2, 3, 4):
            raise ValueError("Supersample must be 1, 2, 3 or 4")
        self.edge_image = wp.empty(tuple(shape), dtype=wp.vec3, device=device) if edge_antialias else None
        self.scale = supersample
        self.shape = tuple(shape)
        self.input_shape = (self.shape[0], self.shape[1]*supersample, self.shape[2]*supersample)
        self.resolved = wp.empty(self.shape, dtype=wp.vec3, device=device) if supersample > 1 else None
        self.pixels = wp.empty((*self.shape, 3), dtype=wp.uint8, device=device)

    def read(self, frame):
        if frame.shape != self.input_shape or frame.dtype != wp.vec3 or frame.device != self.pixels.device:
            raise ValueError('RGB readback requires matching device vec3 image dimensions')
        if self.resolved is not None:
            wp.launch(_resolve, dim=self.shape, inputs=[frame, self.resolved, self.scale], device=self.pixels.device)
            frame = self.resolved
        if self.edge_image is not None:
            wp.launch(_edge_resolve, dim=self.shape, inputs=[frame, self.edge_image], device=self.pixels.device)
            frame = self.edge_image
        wp.launch(_quantize, dim=(*self.shape, 3), inputs=[frame, self.pixels], device=self.pixels.device)
        return self.pixels.numpy()

