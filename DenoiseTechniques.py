from scipy.cluster.hierarchy import weighted
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
import numpy as np
import pywt, cv2

# The implementation
'''
    Shrink method can be one of: BayesShrink, VisuShrink, Sure or NeighborVariation
'''


def dwt_denoise(data, wavelet='db8', level=1, mode='soft', shrinkage='NeighborVariation', color=0):
    mult = 255. if np.max(data) > 1 else 1.
    data = data / mult
    if mode == 'hard':
        denoised = mult * denoise_wavelet(data, wavelet=wavelet, mode=mode, wavelet_levels=level)
    if shrinkage in ['BayesShrink', 'VisuShrink']:
        denoised = mult * denoise_wavelet(data, wavelet=wavelet, mode='soft', method=shrinkage, wavelet_levels=level)
    elif shrinkage == 'Sure':
        raise ValueError('Not implemented')
    # Local variance denoise
    elif shrinkage == 'NeighborVariation':
        # If a color image is received, denoise for each channel
        if len(data.shape) == 3 and data.shape[-1] == 3:
            data = cv2.split(data)
        # For grayscale images, create a single band
        elif len(data.shape) == 2:
            data = [data]
        # If shape is not (x,y,3), (3,x,y) or (x,y) I don't know what to do
        else:
            raise ValueError('Unknown data format')
        dband = []
        for band in data:
            dwt = pywt.wavedec2(band, wavelet=wavelet, level=level)
            dwt2 = dwt_neighvar_shrink(dwt, W=[3, 5, 7, 9])
            dband.append(pywt.waverec2(dwt2, wavelet=wavelet))

        if len(data) == 3:
            denoised = mult * cv2.merge(dband)
        else:
            denoised = mult * dband[0]
    return denoised


'''
'''


def tv_denoise(img, weight=.1):
    mult = 255. if np.max(img) > 1 else 1.
    img = img / mult
    weight = weight / mult
    if len(img.shape) == 3:
        return (mult * denoise_tv_chambolle(img, weight=weight, multichannel=True))
    else:
        return (mult * denoise_tv_chambolle(img, weight=weight))


# def sure_shrink(data):

# Applied to coefficients
def dwt_neighvar_shrink(dwt, W=[3], s2=.5):
    # For every coefficient matrix
    dwt_est = [dwt[0]]
    for cdwt in dwt[1:]:
        cdwt_est = []
        for X in cdwt:
            S2 = np.ones(X.shape) * np.inf
            for w in W:
                S2 = np.minimum(S2, neighvar(X, w, s2))
                # new estimation
            Xbar = X * S2 / (S2 + s2)
            cdwt_est.append(Xbar)
        dwt_est.append(cdwt_est)
    return dwt_est


# Variance estimation
def neighvar(X, w, s2=.5):
    # Variance
    Xmean = cv2.blur(X, (w, w), borderType=cv2.BORDER_REFLECT_101)
    Xvar = cv2.blur(X ** 2, (w, w), borderType=cv2.BORDER_REFLECT_101) - Xmean ** 2
    Xvarmin = np.maximum(np.zeros(Xvar.shape), Xvar - s2)
    # Estimation
    return Xvarmin


def tv_block_denoise(data, bsize, weight=.5):
    h, w = data.shape
    hh, ww, = h // bsize, w // bsize
    D = np.zeros_like(data)
    for i in range(hh):
        for j in range(ww):
            h1, h2 = i * bsize, min((i + 1) * bsize, h)
            w1, w2 = j * bsize, min((j + 1) * bsize, w)
            B = data[h1: h2, w1: w2]
            D[h1: h2, w1: w2] = tv_denoise(B, weight)
    return D


def bilateral_denoise(data):
    return data - denoise_bilateral(data, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)

