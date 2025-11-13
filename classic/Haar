import numpy as np
from PIL import Image

def pad_even(img):
    rows, cols = img.shape[:2]
    if rows % 2 != 0:
        img = np.vstack([img, img[-1:, :]])
    if cols % 2 != 0:
        img = np.hstack([img, img[:, -1:]])
    return img


def dwt1d(signal):
    L = len(signal)
    output = np.zeros_like(signal, dtype=float)
    half = L // 2
    for i in range(half):
        output[i] = (signal[2*i] + signal[2*i+1]) / 2
        output[half+i] = (signal[2*i] - signal[2*i+1]) / 2
    return output

def idwt1d(coeffs):
    L = len(coeffs)
    half = L // 2
    output = np.zeros_like(coeffs, dtype=float)
    for i in range(half):
        output[2*i]   = coeffs[i] + coeffs[half+i]
        output[2*i+1] = coeffs[i] - coeffs[half+i]
    return output

def dwt2d(img):
    temp = np.apply_along_axis(dwt1d, 1, img)
    temp = np.apply_along_axis(dwt1d, 0, temp)
    return temp

def idwt2d(coeffs):
    temp = np.apply_along_axis(idwt1d, 0, coeffs)
    temp = np.apply_along_axis(idwt1d, 1, temp)
    return temp


def super_resolve(img, scale=2):
    coeffs = dwt2d(img)

    rows, cols = img.shape
    half_r, half_c = rows // 2, cols // 2

    LL = coeffs[:half_r, :half_c]
    LH = coeffs[:half_r, half_c:]
    HL = coeffs[half_r:, :half_c]
    HH = coeffs[half_r:, half_c:]

    
    LL_up = np.kron(LL, np.ones((scale, scale)))
    LH_up = np.kron(LH, np.ones((scale, scale)))
    HL_up = np.kron(HL, np.ones((scale, scale)))
    HH_up = np.kron(HH, np.ones((scale, scale)))

    h_min = min(LL_up.shape[0], LH_up.shape[0], HL_up.shape[0], HH_up.shape[0])
    w_min = min(LL_up.shape[1], LH_up.shape[1], HL_up.shape[1], HH_up.shape[1])

    LL_up = LL_up[:h_min, :w_min]
    LH_up = LH_up[:h_min, :w_min]
    HL_up = HL_up[:h_min, :w_min]
    HH_up = HH_up[:h_min, :w_min]

    new_rows, new_cols = h_min * 2, w_min * 2
    new_coeffs = np.zeros((new_rows, new_cols))

    new_coeffs[:h_min, :w_min] = LL_up
    new_coeffs[:h_min, w_min:w_min*2] = LH_up
    new_coeffs[h_min:h_min*2, :w_min] = HL_up
    new_coeffs[h_min:h_min*2, w_min:w_min*2] = HH_up

    img_up = idwt2d(new_coeffs)
    return np.clip(img_up, 0, 255).astype(np.uint8)



def pad_to_even(img):
   
    rows, cols = img.shape[:2]
    pad_bottom = rows % 2   
    pad_right  = cols % 2

    if pad_bottom == 0 and pad_right == 0:
        return img  

    if img.ndim == 2:
        pad_width = ((0, pad_bottom), (0, pad_right))
    elif img.ndim == 3:
        pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))
    else:
        raise ValueError("img deve ser 2D ou 3D")

    img_padded = np.pad(img, pad_width, mode='reflect')
    return img_padded

def process_image(path_in, path_out, scale=2, mode="L"):
   
    img = Image.open(path_in).convert(mode)
    arr = np.array(img, dtype=float)

    if mode == "L":
        arr_sr = super_resolve(arr, scale)
    elif mode == "RGB":
        channels = []
        for i in range(3):
            ch_sr = super_resolve(arr[:, :, i], scale)
            channels.append(ch_sr)
        arr_sr = np.stack(channels, axis=2)
    else:
        raise ValueError("Modo inválido. Use 'L' ou 'RGB'.")

    Image.fromarray(arr_sr).save(path_out)


if __name__ == "__main__":
    process_image("input/exemplo.png", "output/finalHaar.png", scale=2, mode="RGB")
