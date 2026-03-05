from PIL import Image
import numpy as np


img1 = Image.open('media/yuri.webp')
img2 = Image.open('media/yuri2.webp')
img3 = Image.open('media/Meeeeeeeee.jpg')
images = [img1, img2, img3]
img_arrays = list(map(np.array, images))
top = 0
shapes = []
for i in img_arrays:
    shapes.append(i.shape)
mshape = max(shapes)
result = np.zeros(mshape, dtype=np.uint8)
for img in reversed(img_arrays):
    h, w = img.shape[:2]
    result[:h, :w] = img
Image.fromarray(result).show()
