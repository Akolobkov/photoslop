import numpy as np
from PIL import Image
def gavr(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)

    t = np.mean(img_arr)
    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    img_shape = img_arr.shape
    img_vec = img_arr.flatten()
    gavr_pic = np.zeros_like(img_vec)
    for i in range(len(img_vec)):
        if img_vec[i]>t:
            gavr_pic[i] = 255
    gavr_pic = gavr_pic.reshape(img_shape)
    gavr_pic = gavr_pic.astype(np.uint8)
    gavr_img = Image.fromarray(gavr_pic)
    return gavr_img
def otsu(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)

    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    img_shape = img_arr.shape
    img_vec = img_arr.flatten()
    img_int = np.round(img_arr).astype(np.uint8)
    counts = np.bincount(img_int.flatten(), minlength=256)
    total_pixels = counts.sum()
    hist = counts / total_pixels
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(np.arange(256) * hist)

    global_mean = cumsum_mean[-1]

    w1 = cumsum[:-1]
    w2 = 1 - w1
    u1 = cumsum_mean[:-1] / w1
    u2 = (global_mean - cumsum_mean[:-1]) / w2
    sig = w1 * w2 * (u1 - u2) ** 2
    sig = np.nan_to_num(sig)

    maxt = np.argmax(sig) + 1
    otsu_pic = np.zeros_like(img_vec)
    for i in range(len(img_vec)):
        if img_vec[i] > maxt:
            otsu_pic[i] = 255
    otsu_pic = otsu_pic.reshape(img_shape)
    otsu_pic = otsu_pic.astype(np.uint8)
    otsu_img = Image.fromarray(otsu_pic)
    return otsu_img
def niblek(img_path, a, k):
    img = Image.open(img_path)
    img_arr = np.array(img)
    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    window_shape = (a, a)
    windows = np.lib.stride_tricks.sliding_window_view(img_arr, window_shape)
    M = np.mean(windows, axis=(2,3))
    M2 = np.mean(windows**2, axis = (2,3))
    D = M2 - M**2
    sig = np.sqrt(D)
    t = M + k*sig
    img_cropped = img_arr[:t.shape[0], :t.shape[1]]
    pic = (img_cropped > t)*255
    pic = pic.astype(np.uint8)
    pic = Image.fromarray(pic)
    return pic
def savuola(img_path, a, k=0.2):
    img = Image.open(img_path)
    img_arr = np.array(img)
    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    window_shape = (a, a)
    windows = np.lib.stride_tricks.sliding_window_view(img_arr, window_shape)
    M = np.mean(windows, axis=(2,3))
    M2 = np.mean(windows**2, axis = (2,3))
    D = M2 - M**2
    sig = np.sqrt(D)
    t = M *(1+ k*(sig/128-1))
    img_cropped = img_arr[:t.shape[0], :t.shape[1]]
    pic = (img_cropped > t) * 255
    pic = pic.astype(np.uint8)
    pic = Image.fromarray(pic)
    return pic
def wolf(img_path, a, k, ag = 0.5):
    img = Image.open(img_path)
    img_arr = np.array(img)
    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
    window_shape = (a, a)
    windows = np.lib.stride_tricks.sliding_window_view(img_arr, window_shape)
    M = np.mean(windows, axis=(2,3))
    M2 = np.mean(windows**2, axis = (2,3))
    D = M2 - M**2
    sig = np.sqrt(np.maximum(D, 0))
    m = np.min(img_arr)
    R = np.max(sig)
    t = (1-0.5) *M + 0.5* m + 0.5*sig * (M - m)/R
    img_cropped = img_arr[:t.shape[0], :t.shape[1]]
    pic = (img_cropped > t) * 255
    pic = pic.astype(np.uint8)
    pic = Image.fromarray(pic)
    return pic


def bradleyrot(img_path, a, k, ag=0.5):
    img = Image.open(img_path)
    img_arr = np.array(img)
    img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])

    h, w = img_arr.shape
    S = np.zeros((h, w), dtype=np.float64)


    for x in range(h):
        for y in range(w):
            S[x][y] = img_arr[x][y]
            if x > 0:
                S[x][y] += S[x - 1][y]
            if y > 0:
                S[x][y] += S[x][y - 1]
            if x > 0 and y > 0:
                S[x][y] -= S[x - 1][y - 1]

    pic = np.zeros((h, w), dtype=np.uint8)
    half_a = a // 2

    for x in range(h):
        for y in range(w):
            x1 = max(0, x - half_a)
            x2 = min(h - 1, x + half_a)
            y1 = max(0, y - half_a)
            y2 = min(w - 1, y + half_a)


            C = (x2 - x1 + 1) * (y2 - y1 + 1)


            total = S[x2][y2]

            if x1 > 0:
                total -= S[x1 - 1][y2]
            if y1 > 0:
                total -= S[x2][y1 - 1]
            if x1 > 0 and y1 > 0:
                total += S[x1 - 1][y1 - 1]

            if img_arr[x][y] * C >= total * (1 - k / 100.0):
                pic[x][y] = 255

    pic = pic.astype(np.uint8)
    pic = Image.fromarray(pic)
    return pic

