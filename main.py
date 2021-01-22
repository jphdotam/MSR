import os
import matplotlib.pyplot as plt

from lib.data import load_real_imaginary_image_gmap, add_noise_to_gmap, add_noise_to_image
from lib.kspace import cut_and_filter_image_in_kspace

DATA_DIR = "../from_hui"
FILTER_DIR = "../kspace_filters"
REAL_PATH = os.path.join(DATA_DIR, "images_for_gmap_real.npy")
IMAG_PATH = os.path.join(DATA_DIR, "images_for_gmap_imag.npy")
GMAP_PATH = os.path.join(DATA_DIR, "gmap_slc_1.npy")

img_complex, img_mag, img_cat, gmap = load_real_imaginary_image_gmap(REAL_PATH, IMAG_PATH, GMAP_PATH)

# Add white noise
gmap_noised = add_noise_to_gmap(gmap, noise_factor=0.25)
img_noised = add_noise_to_image(img_complex, gmap_noised)

# Low res
img_lowres, img_lowres_mag = cut_and_filter_image_in_kspace(img_noised, 64, FILTER_DIR)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_mag[:, :, 0], cmap='gray')
axes[0].set_title("orig")
axes[0].axis('off')

axes[1].imshow(img_lowres_mag[:, :, 0], cmap='gray')
axes[1].set_title("low")
axes[1].axis('off')

plt.show()
