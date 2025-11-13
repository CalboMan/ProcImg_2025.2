import numpy as np
from PIL import Image

def mexican_hat(size=7, sigma=1.0):
    kernel = np.zeros((size, size), dtype=float)
    center = size // 2
    soma = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            r2 = x*x + y*y
            val = (2 - (r2 / sigma**2)) * np.exp(-r2 / (2 * sigma**2))
            kernel[i, j] = val
            soma += val

    media = soma / (size * size)
    for i in range(size):
        for j in range(size):
            kernel[i, j] -= media

    return kernel

def conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=float)

    for i in range(h):
        for j in range(w):
            padded[i + pad_h, j + pad_w] = img[i, j]

    for i in range(pad_h):
        for j in range(w):
            padded[i, j + pad_w] = img[pad_h - i, j]            
            padded[h + pad_h + i, j + pad_w] = img[h - 1 - i, j]  
    for i in range(h + 2*pad_h):
        for j in range(pad_w):
            padded[i, j] = padded[i, pad_w + (pad_w - j)]          
            padded[i, w + pad_w + j] = padded[i, w + pad_w - 1 - j] 

    result = np.zeros((h, w), dtype=float)

    for i in range(h):
        for j in range(w):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += padded[i + ki, j + kj] * kernel[ki, kj]
            result[i, j] = acc

    return result

def upsample(img, scale=2):
    h, w = img.shape
    new_h, new_w = h * scale, w * scale
    up = np.zeros((new_h, new_w), dtype=float)

    for i in range(h):
        for j in range(w):
            for si in range(scale):
                for sj in range(scale):
                    up[i*scale + si, j*scale + sj] = img[i, j]

    return up

def super_resolve(img, scale=2, sigma=1.0):
    kernel = mexican_hat(size=7, sigma=sigma)

    edges = conv2d(img, kernel)

    base = img - edges

    up_base = upsample(base, scale)
    up_edges = upsample(edges, scale)

    for i in range(up_edges.shape[0]):
        for j in range(up_edges.shape[1]):
            up_edges[i, j] *= 1.3  # alfa = 1.3

    sr = np.zeros_like(up_base)
    for i in range(sr.shape[0]):
        for j in range(sr.shape[1]):
            sr[i, j] = up_base[i, j] + up_edges[i, j]

    for i in range(sr.shape[0]):
        for j in range(sr.shape[1]):
            if sr[i, j] < 0:
                sr[i, j] = 0
            elif sr[i, j] > 255:
                sr[i, j] = 255

    return sr.astype(np.uint8)

def process_image(path_in, path_out, scale=2, mode="L"):
    img = Image.open(path_in).convert(mode)
    arr = np.array(img, dtype=float)

    if mode == "L":
        arr_sr = super_resolve(arr, scale)
    else:
        channels = []
        for i in range(3):
            ch_sr = super_resolve(arr[:, :, i], scale)
            channels.append(ch_sr)
        arr_sr = np.stack(channels, axis=2)

    Image.fromarray(arr_sr).save(path_out)

if __name__ == "__main__":
    process_image("input/exemplo.png", "output/finalMexican.png", scale=2, mode="RGB")
