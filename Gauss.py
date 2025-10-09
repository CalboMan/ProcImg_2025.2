import numpy as np
from PIL import Image

def gaussian(size=7, sigma=1.0):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.zeros((size, size), dtype=float)
    soma = 0.0

    for i in range(size):
        for j in range(size):
            x = ax[i]
            y = ax[j]
            val = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i, j] = val
            soma += val

    kernel /= soma
    return kernel

def conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=float)

    for i in range(h):
        for j in range(w):
            padded[i + pad_h, j + pad_w] = img[i, j]

    for i in range(pad_h):
        padded[i, pad_w:-pad_w] = padded[2 * pad_h - i, pad_w:-pad_w]
        padded[-(i+1), pad_w:-pad_w] = padded[-(2 * pad_h + 1 - i), pad_w:-pad_w]
    for j in range(pad_w):
        padded[:, j] = padded[:, 2 * pad_w - j]
        padded[:, -(j+1)] = padded[:, -(2 * pad_w + 1 - j)]

    result = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc += padded[i + ki, j + kj] * kernel[ki, kj]
            result[i, j] = acc

    return result

def upsample(img, scale):
    h, w = img.shape
    up = np.zeros((h * scale, w * scale), dtype=float)
    for i in range(h):
        for j in range(w):
            for si in range(scale):
                for sj in range(scale):
                    up[i * scale + si, j * scale + sj] = img[i, j]
    return up

def super_resolve(img, scale=2, sigma=1.0):
    kernel = gaussian(size=7, sigma=sigma)
    smooth = conv2d(img, kernel)

    detail = img - smooth

    up_smooth = upsample(smooth, scale)
    up_detail = upsample(detail, scale) * 1.5 

    sr = up_smooth + up_detail
    return np.clip(sr, 0, 255).astype(np.uint8)

def process_image(path_in, path_out, scale=2, mode="L"):
    img = Image.open(path_in).convert(mode)
    arr = np.array(img, dtype=float)

    if mode == "L":
        arr_sr = super_resolve(arr, scale)
    else:
        channels = []
        for i in range(3):
            print(f"Processando canal {i+1}/3...")
            sr_c = super_resolve(arr[:, :, i], scale)
            channels.append(sr_c)
        arr_sr = np.stack(channels, axis=2)

    Image.fromarray(arr_sr).save(path_out)
    print(f"Imagem salva em {path_out}")

if __name__ == "__main__":
    process_image("input/exemplo.png", "output/finalGaussiana.png", scale=2, mode="RGB")

