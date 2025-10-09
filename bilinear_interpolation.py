from PIL import Image
import numpy as np

def bilinear_interpolate(img, x, y):
    h, w = img.shape[:2]
    
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    
    dx = x - x0
    dy = y - y0
    
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    
    top_left = img[y0, x0] 
    top_right = img[y0, x1]  
    bottom_left = img[y1, x0] 
    bottom_right = img[y1, x1] 

    R0 = top_left * (1 - dx) + top_right * dx  
    R1 = bottom_left * (1 - dx) + bottom_right * dx  
    
    result = R0 * (1 - dy) + R1 * dy
    
    return result

def bilinear_upscale(image_path, scale_factor=2):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    h, w = img_array.shape[:2]
    
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    upscaled = np.zeros((new_h, new_w, img_array.shape[2]))

    for i in range(new_h):
        if i % 100 == 0:
            print(f"Progress: {i}/{new_h} rows")
        
        for j in range(new_w):
            x = j / scale_factor
            y = i / scale_factor
            
            upscaled[i, j] = bilinear_interpolate(img_array, x, y)
    
    upscaled = np.clip(upscaled, 0, 255)
    
    return upscaled.astype(np.uint8)

if __name__ == "__main__":
    img = bilinear_upscale("input/exemplo.png", scale_factor=4)

    resized_img = Image.fromarray(img)

    resized_img.save("output/finalBilinear.png")
