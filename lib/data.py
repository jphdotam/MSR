import numpy as np


def generate_noise(img_cat, noise_factor=0.25):
    height, width = img_cat.shape[:2]
    mean_signal = np.mean(img_cat)
    noise = (noise_factor * mean_signal) * (np.random.rand(height, width) + (1j * np.random.rand(height, width)))
    return noise


def add_noise_to_gmap(gmap, noise_factor):
    noise = generate_noise(gmap, noise_factor=noise_factor)
    gmap_noised = noise * gmap
    return gmap_noised


def add_noise_to_image(img_complex, gmap_noised):
    img_noised = img_complex + gmap_noised
    return img_noised
