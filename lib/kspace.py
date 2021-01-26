import os
import math
import numpy as np


def cut_and_filter_image_in_kspace(img_noised, n_lines_kspace, filter_dir):
    rows_in, cols_in = img_noised.shape[:2]
    line_from = (cols_in //2) - (n_lines_kspace//2)
    line_to = line_from + n_lines_kspace

    # FFT
    kspace = fft2c(img_noised)
    kspace_cut = kspace[:, line_from:line_to]

    # Filter kspace
    filter_row = np.load(os.path.join(filter_dir, f"filter_len_{rows_in}.npy"))
    filter_n_lines = np.load(os.path.join(filter_dir, f"filter_len_{n_lines_kspace}.npy"))

    # kspace_cut_filtered = np.dstack(
    #     [filter_kspace(kspace_cut[:, :, i], filter_row, filter_n_lines) for i in range(kspace_cut.shape[2])])
    kspace_cut_filtered = filter_kspace(kspace_cut, filter_row, filter_n_lines)
    kspace_small = np.zeros_like(kspace)
    kspace_small[:, line_from:line_to] = kspace_cut_filtered

    # iFFT
    img_lowres = ifft2c(kspace_small)
    img_lowres_mag = np.abs(img_lowres)
    return img_lowres, img_lowres_mag


def filter_kspace(kspace, filter_fe=None, filter_pe=None):
    cols, lins = kspace.shape

    if filter_fe is None:
        filter_fe = np.ones((cols, 1))
    r = 1 / math.sqrt(1 / cols * np.sum(filter_fe * filter_fe))
    filter_fe = filter_fe * r

    if filter_pe is None:
        filter_pe = np.ones((lins, 1))
    r = 1 / math.sqrt(1 / lins * np.sum(filter_pe * filter_pe))
    filter_pe = filter_pe * r

    filter_2d = np.matmul(filter_fe, filter_pe.T)

    kspace_filtered = filter_2d * kspace

    return kspace_filtered

def fft2c(x):
    fctr = x.shape[0] * x.shape[1]
    return 1 / math.sqrt(fctr) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(x):
    fctr = x.shape[0] * x.shape[1]
    return math.sqrt(fctr) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
