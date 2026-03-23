from PIL import Image
import numpy as np
import os


def layer_images(images, alphas):

    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    alpha_bg = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    c_bg = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        alpha_src = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        c_src = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        alpha_src[:h, :w] = img[:, :, 3] / 255.0
        c_src[:h, :w, :] = img[:, :, :3] / 255.0

        alpha_src = alpha_src * alpha

        alpha_out = alpha_src + alpha_bg * (1 - alpha_src)

        mask = alpha_out > 0
        c_out = np.zeros_like(c_src)

        c_out[mask] = (
                              c_src[mask] * alpha_src[mask, np.newaxis] +
                              c_bg[mask] * alpha_bg[mask, np.newaxis] * (1 - alpha_src[mask, np.newaxis])
                      ) / alpha_out[mask, np.newaxis]

        c_out[~mask] = c_bg[~mask]

        alpha_bg = alpha_out
        c_bg = c_out

    # Создаем итоговое изображение
    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (c_bg * 255).astype(np.uint8)
    result[:, :, 3] = (alpha_bg * 255).astype(np.uint8)
    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img

def sum_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))


    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]


    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        # Получаем альфа-канал и цвет текущего изображения
        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0

        # Суммируем с учетом альфа-каналов
        # Обновляем альфа: сумма альф (но не больше 1)
        new_alpha = result_alpha + current_alpha
        new_alpha = np.clip(new_alpha, 0, 1)

        # Обновляем цвет: взвешенная сумма по альфа-каналам
        mask = new_alpha > 0
        result_color_new = np.zeros_like(result_color)

        if np.any(mask):
            # Взвешенное суммирование цветов с учетом их альф
            result_color_new[mask] = (
                    (result_color[mask] * result_alpha[mask, np.newaxis] +
                     current_color[mask] * current_alpha[mask, np.newaxis]) /
                    new_alpha[mask, np.newaxis]
            )

        result_color_new[~mask] = result_color[~mask]

        result_alpha = new_alpha
        result_color = result_color_new

    # Создаем итоговое изображение
    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    # Сохраняем результат
    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img



def mul_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0

        # Для первого изображения просто копируем
        if i == 0:
            result_alpha = current_alpha
            result_color = current_color
            continue

        new_alpha = result_alpha + current_alpha
        new_alpha = np.clip(new_alpha, 0, 1)

        result_color_new = np.zeros_like(result_color)

        mask_both = (result_alpha > 0) & (current_alpha > 0)
        if np.any(mask_both):
            result_color_new[mask_both] = (
                    result_color[mask_both] * current_color[mask_both]
            )

        mask_current_only = (result_alpha == 0) & (current_alpha > 0)
        if np.any(mask_current_only):
            result_color_new[mask_current_only] = current_color[mask_current_only]

        mask_result_only = (result_alpha > 0) & (current_alpha == 0)
        if np.any(mask_result_only):
            result_color_new[mask_result_only] = result_color[mask_result_only]

        result_color = result_color_new
        result_alpha = new_alpha

    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img
def sub_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))


    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]


    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0

        if i == 0:
            result_alpha = current_alpha
            result_color = current_color
            continue

        new_alpha = result_alpha.copy()

        result_color_new = result_color.copy()

        mask_both = (result_alpha > 0) & (current_alpha > 0)
        if np.any(mask_both):
            result_color_new[mask_both] = np.abs(
                result_color[mask_both] - current_color[mask_both]
            )


        result_color = result_color_new
        result_alpha = new_alpha

    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img
def max_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0


        new_alpha = np.maximum(result_alpha, current_alpha)
        new_alpha = np.clip(new_alpha, 0, 1)

        mask = (new_alpha > 0) & (current_alpha >0)
        result_color_new = result_color.copy()

        if np.any(mask):
            result_color_new[mask] = np.maximum(result_color[mask], current_color[mask])
        result_color_new[~mask] = result_color[~mask]

        result_alpha = new_alpha
        result_color = result_color_new
    mask_current_only = (result_alpha == 0) & (current_alpha > 0)
    if np.any(mask_current_only):
        result_color_new[mask_current_only] = current_color[mask_current_only]
    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img

def geom_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0

        if i == 0:
            result_alpha = current_alpha
            result_color = current_color
            continue

        new_alpha = result_alpha + current_alpha
        new_alpha = np.clip(new_alpha, 0, 1)

        result_color_new = np.zeros_like(result_color)

        mask_both = (result_alpha > 0) & (current_alpha > 0)
        if np.any(mask_both):
            result_color_new[mask_both] = (
                    np.sqrt(result_color[mask_both] * current_color[mask_both])
            )

        mask_current_only = (result_alpha == 0) & (current_alpha > 0)
        if np.any(mask_current_only):
            result_color_new[mask_current_only] = current_color[mask_current_only]

        mask_result_only = (result_alpha > 0) & (current_alpha == 0)
        if np.any(mask_result_only):
            result_color_new[mask_result_only] = result_color[mask_result_only]

        result_color = result_color_new
        result_alpha = new_alpha

    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    # Сохраняем результат
    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img
def sr_images(images, alphas):
    img_arrays = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_arrays.append(np.array(img))

    shapes = [img.shape for img in img_arrays]
    mshape = [
        max(shape[0] for shape in shapes),
        max(shape[1] for shape in shapes)
    ]

    result_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
    result_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

    for i in range(len(img_arrays)):
        img = img_arrays[i]
        alpha = alphas[i] if i < len(alphas) else 1.0

        h, w = img.shape[:2]

        current_alpha = np.zeros((mshape[0], mshape[1]), dtype=np.float32)
        current_color = np.zeros((mshape[0], mshape[1], 3), dtype=np.float32)

        current_alpha[:h, :w] = img[:, :, 3] / 255.0 * alpha
        current_color[:h, :w, :] = img[:, :, :3] / 255.0

        new_alpha = result_alpha + current_alpha
        new_alpha = np.clip(new_alpha, 0, 1)

        mask = new_alpha > 0
        result_color_new = np.zeros_like(result_color)

        if np.any(mask):
            result_color_new[mask] = (
                    (result_color[mask] * result_alpha[mask, np.newaxis] +
                     current_color[mask] * current_alpha[mask, np.newaxis]) /
                    (new_alpha[mask, np.newaxis] * 2)
            )

        result_color_new[~mask] = result_color[~mask]

        result_alpha = new_alpha
        result_color = result_color_new

    result_color = np.clip(result_color, 0, 1)
    result_alpha = np.clip(result_alpha, 0, 1)

    result = np.zeros((mshape[0], mshape[1], 4), dtype=np.uint8)
    result[:, :, :3] = (result_color * 255).astype(np.uint8)
    result[:, :, 3] = (result_alpha * 255).astype(np.uint8)

    # Сохраняем результат
    img = Image.fromarray(result)
    result_filename = 'result.png'
    img.save(result_filename, 'PNG')
    return img
