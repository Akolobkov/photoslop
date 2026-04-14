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
    gavr_img = Image.fromarray(gavr_pic)
    gavr_img.show()
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
    gavr_pic = np.zeros_like(img_vec)
    for i in range(len(img_vec)):
        if img_vec[i] > maxt:
            gavr_pic[i] = 255
    gavr_pic = gavr_pic.reshape(img_shape)
    gavr_img = Image.fromarray(gavr_pic)
    gavr_img.show()
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
    pic = img_cropped > t
    pic = Image.fromarray(pic)
    pic.show()
def savuola(img_path, a, k):
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
    pic = img_cropped > t
    pic = Image.fromarray(pic)
    pic.show()
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
    print(m, R, t)
    img_cropped = img_arr[:t.shape[0], :t.shape[1]]
    pic = img_cropped > t
    pic = Image.fromarray(pic)
    pic.show()
niblek('pic/IMG_20260414_094345_560.jpg', 15, -0.2)

