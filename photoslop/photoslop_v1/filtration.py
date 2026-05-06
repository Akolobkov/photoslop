import numpy as np
from PIL import Image
def quickselect(matrix):

    def partition(arr, left, right, pivotIndex):
        pivotValue = arr[pivotIndex]
        arr[pivotIndex], arr[right] = arr[right], arr[pivotIndex]
        storeIndex = left
        for i in range(left, right-1):
            if arr[i]  < pivotValue:
                arr[storeIndex], arr[i] = arr[i], arr[storeIndex]
                storeIndex += 1
        arr[right], arr[storeIndex]  =  arr[storeIndex], arr[right]
        return storeIndex
    def select(arr, left, right, k):
        if left == right:
            return arr[left]
        pivotIndex = left + np.random.randint(0, right - left)
        pivotIndex = partition(arr, left, right, pivotIndex)
        if k == pivotIndex:
            return arr[k]
        elif k  < pivotIndex:
            return select(arr, left, pivotIndex - 1, k)
        else:
            return select(arr, pivotIndex + 1, right, k)
    arr = matrix.flatten()
    return select(arr, 0, len(arr)-1, len(arr)//2)
def gauss_matrix(sig, r):
    s = 0
    sig_sqr = 2.0 * sig * sig
    pi_siq_sqr = sig_sqr * np.pi
    i = np.arange(-r, r + 1)
    j = np.arange(-r, r + 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    g_matrix = 1.0 / pi_siq_sqr * np.exp(-1.0 * (ii * ii + jj * jj) / sig_sqr)
    s = np.sum(g_matrix)
    return g_matrix, s

def linearFromInputtedMatrix(img_path, matrix):
    img = Image.open(img_path)
    img_arr = np.array(img)
    h, w, с = img_arr.shape
    window_shape = np.shape(matrix)
    pad_h = window_shape[0] // 2
    pad_w = window_shape[1] // 2
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    ans = []
    for arr in (R, G, B):
        padded = np.pad(arr, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape)
        result = windows * matrix
        result = np.sum(result, axis =  (-2, -1))
        ans.append(result)
    img = np.stack((ans[0], ans[1], ans[2]), axis=2)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img
def linearFromGaussianMatrix(img_path, sig, r):
    img = Image.open(img_path)
    img_arr = np.array(img)
    h, w, с = img_arr.shape
    matrix, s = gauss_matrix(sig, r)
    window_shape = np.shape(matrix)
    pad_h = window_shape[0] // 2
    pad_w = window_shape[1] // 2
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    ans = []
    for arr in (R, G, B):
        padded = np.pad(arr, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape)
        result = windows * matrix
        result = np.sum(result, axis =  (-2, -1))
        ans.append(result)
    img = np.stack((ans[0], ans[1], ans[2]), axis=2)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img
def median(img_path, window_shape):
    window_shape= list(map(int, window_shape))
    img = Image.open(img_path)
    img_arr = np.array(img)
    h, w, с = img_arr.shape
    pad_h = window_shape[0] // 2
    pad_w = window_shape[1] // 2
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    ans = []
    for arr in (R, G, B):
        padded = np.pad(arr, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape)
        filtered = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                window = windows[i, j].flatten()
                filtered[i, j] = quickselect(window)
        ans.append(filtered)


    img = np.stack((ans[0], ans[1], ans[2]), axis=2)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img