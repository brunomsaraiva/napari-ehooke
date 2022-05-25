import napari
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass
from skimage.transform import EuclideanTransform, warp
from skimage.filters import threshold_isodata, threshold_otsu
from napari.utils.notifications import show_info
from scipy import ndimage as ndi
from skimage.morphology import closing


def compute_binary_mask(img, img_type, algorithm, mask_closing, fill_holes):
    if algorithm == "Isodata":
        mask = img > threshold_isodata(img)

    elif algorithm == "Otsu":
        mask = img > threshold_otsu(img)
    else:
        show_info("Not a valid thresholding algorithm!")
        return

    if img_type == "Phase":
        mask = 1 - mask

    if mask_closing > 0:
        closing_matrix = np.ones((int(mask_closing), int(mask_closing)))
        mask = closing(mask, closing_matrix)
        mask = 1 - closing(1 - mask, closing_matrix)
    
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    return mask

def align_img_to_mask(img, mask):
    corr = fftconvolve(mask, img[::-1, ::-1])
    deviation = np.unravel_index(np.argmax(corr), corr.shape)
    cm = center_of_mass(np.ones(corr.shape))
    best = np.subtract(deviation, cm)
    dy, dx = best
    final_matrix = EuclideanTransform(rotation=0, translation=(dx, dy))
    return warp(img, final_matrix.inverse,preserve_range=True)

