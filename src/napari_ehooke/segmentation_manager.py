import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.util import img_as_int, img_as_uint
from skimage.exposure import rescale_intensity
from skimage.segmentation import watershed
from skimage.morphology import erosion, dilation
from skimage.color import rgb2gray


def compute_distance_peaks(mask):
    distance = ndi.distance_transform_edt(mask)
    mindist = 3
    minvalue = 3

    centers_coords = peak_local_max(distance,
                             min_distance=mindist,
                             threshold_abs=minvalue,
                             exclude_border=True,
                             indices=True)
    centers = np.zeros(distance.shape, dtype=bool)
    centers[tuple(centers_coords.T)] = True
    markers, _ = ndi.label(centers)

    return markers

def segment_single_cells(mask):
    markers = compute_distance_peaks(mask)
    markers = np.array(markers)

    distance = ndi.morphology.distance_transform_edt(mask)
    distance = np.array(distance)

    labels = watershed(-distance, markers, mask=mask)

    return labels

