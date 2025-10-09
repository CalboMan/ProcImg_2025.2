import numpy as np
from PIL import Image

def cubic_kernel(x, a=-0.5):
    x = abs(x)
    if x <= 1:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1
    elif x < 2:
        return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a
    else:
        return 0

def get_bicubic_weights(dx, dy, a=-0.5):
    weights = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            dist_x = dx - (i - 1)
            dist_y = dy - (j - 1)
            
            weights[i, j] = cubic_kernel(dist_x, a) * cubic_kernel(dist_y, a)
    
    return weights

def bicubic_interpolate(img, x, y, a=-0.5):
    h, w = img.shape[:2]
    
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0
    
    weights = get_bicubic_weights(dx, dy, a)
    
    if len(img.shape) == 3:
        result = np.zeros(img.shape[2])
    else:
        result = 0.0
    
    for i in range(4):
        for j in range(4):
            xi = x0 + i - 1
            yj = y0 + j - 1
            
            xi = max(0, min(w - 1, xi))
            yj = max(0, min(h - 1, yj))
            
            result += weights[i, j] * img[yj, xi]
    
    return result

def bicubic_upscale(image_path, scale_factor=2, a=-0.5):
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)
    
    h, w = img_array.shape[:2]
    
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    if len(img_array.shape) == 3:
        upscaled = np.zeros((new_h, new_w, img_array.shape[2]), dtype=np.float32)
    else:
        upscaled = np.zeros((new_h, new_w), dtype=np.float32)
    
    for i in range(new_h):
        if i % 100 == 0:
            print(f"Progress: {i}/{new_h} rows")
        
        for j in range(new_w):
            x = j / scale_factor
            y = i / scale_factor
            
            upscaled[i, j] = bicubic_interpolate(img_array, x, y, a)
    
    upscaled = np.clip(upscaled, 0, 255)
    
    return upscaled.astype(np.uint8)

if __name__ == "__main__":
    input_path = "input/exemplo.png"
    output_path = "output/finalBicubica.png"

    result = bicubic_upscale(input_path, scale_factor=4, a=-0.5)
    
    Image.fromarray(result).save(output_path)
    
    original = np.array(Image.open(input_path))
