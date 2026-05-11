from PIL import Image
import numpy as np
import os
import json
import pickle
from functools import wraps
def cache_to_file(cache_file):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    return decorator
def G(u, xarr):
    N = len(xarr)
    s = 0
    for k in range(N):
        s += xarr[k] * np.exp(1j * (-2*np.pi * u* k / N))
    return  1/N * s
def xu(u, Gs):
    N = len(Gs)
    s = 0
    for k in range(N):
        s += Gs[k] * np.exp(1j * (2*np.pi * u* k / N))
    return s
def onedimDFT(xarr):
    Gs = np.zeros_like(xarr, dtype=complex)
    for u in range(len(xarr)):
        Gs[u] = G(u, xarr)
    return Gs
def backardonedimDFT(Gs):
    xarr = np.zeros_like(Gs)
    N = len(xarr)
    for u in range(N):
        xarr[u] = xu(u, Gs)
    return xarr
@cache_to_file('Twodim_result.pkl')
def twodimDFT(img_path):
    img = Image.open(img_path)
    img_arr = np.array(img)
    M, N, c = img_arr.shape
    print(M, N)
    R, G, B = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    Gs = []
    for arr in (R, G, B):
        signs = (-1) ** (np.arange(M)[:, None] + np.arange(N))
        arr = arr * signs
        X1 = np.apply_along_axis(onedimDFT, axis = 1, arr = arr)
        G = np.apply_along_axis(onedimDFT, axis = 0, arr=X1)
        Gs.append(G)
    return Gs
def backwardtwodimDFT(G):
    M, N = G.shape
    X1 = np.apply_along_axis(backardonedimDFT, axis = 1, arr = G)
    X = np.apply_along_axis(backardonedimDFT, axis=0, arr=X1)
    signs = (-1) ** (np.arange(M)[:, None] + np.arange(N))
    X = X * signs
    return X
def visualize_furie(Gs):
    ans = []
    for G in Gs:
        I = np.abs(G)*70
        ans.append(I)
    img = np.stack((ans[0], ans[1], ans[2]), axis=2)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
def lowfreqfilter(G, r):
    M, N = G.shape
    cx, cy = (N - 1) / 2, (M - 1) / 2
    y, x = np.ogrid[:M, :N]
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = distances <= r
    G[~mask] = 0
    return G
def highfreqfilter(G, r):
    M, N = G.shape
    cx, cy = (N - 1) / 2, (M - 1) / 2
    y, x = np.ogrid[:M, :N]
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = distances <= r
    G[mask] = 0
    return G
Gs = twodimDFT('pic/images.jfif')
ans = []
for G in Gs:
    G = highfreqfilter(G, 50)
    ans.append(backwardtwodimDFT(G))
img = np.stack((ans[0], ans[1], ans[2]), axis=2)
img = np.clip(img, 0, 255)
img = img.astype(np.uint8)
img = Image.fromarray(img)
img.show()