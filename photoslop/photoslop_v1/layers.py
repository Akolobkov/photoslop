from PIL import Image
import numpy as np
import os

# Проверка существования файлов
files = ['media/yuri.webp', 'media/yuri2.webp', 'media/Meeeeeeeee.jpg']
for file in files:
    if not os.path.exists(file):
        print(f"Файл {file} не найден!")
        exit()

img1 = Image.open('media/yuri.webp')
img2 = Image.open('media/yuri2.webp')
img3 = Image.open('media/Meeeeeeeee.jpg')
images = [img1, img2, img3]
img_arrays = []

for img in images:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_arrays.append(np.array(img))

alphas = [0.7, 0.5, 1]

shapes = [img.shape for img in img_arrays]
mshape = [max(shape[0] for shape in shapes),
          max(shape[1] for shape in shapes)]

result = np.zeros((mshape[0], mshape[1], 4), dtype=np.float32)

alpha_bg = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
c_bg = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

for i in range(len(img_arrays) - 1, -1, -1):
    img = img_arrays[i]

    h, w = img.shape[:2]

    alpha_src = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    c_src = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)
    alpha_src[:h, :w] = img[:, :, 3] / 255.0
    c_src[:h, :w, :] = img[:, :, :3] / 255.0

    alpha_src = alpha_src * alphas[i]
    alpha_out = alpha_src + alpha_bg * (1 - alpha_src)

    mask = alpha_out > 0
    c_out = np.zeros_like(c_src)

    c_out[mask] = (c_src[mask] * alpha_src[mask, np.newaxis] +
                   c_bg[mask] * alpha_bg[mask, np.newaxis] * (1 - alpha_src[mask, np.newaxis])) / alpha_out[
                      mask, np.newaxis]
    c_out[~mask] = c_bg[~mask]

    alpha_bg = alpha_out
    c_bg = c_out

result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
result[:, :, :3] = (c_bg * 255).astype(np.uint8)
result[:, :, 3] = (alpha_bg * 255).astype(np.uint8)

print(result.shape)
print("Минимальное значение:", result.min())
print("Максимальное значение:", result.max())

# Сохраняем результат
output_image = Image.fromarray(result)
output_image.save('media/result.png')
print("Результат сохранен в media/result.png")