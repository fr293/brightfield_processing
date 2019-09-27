from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, threshold_local, threshold_sauvola
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max
import numpy as np
from matplotlib.patches import Circle
from scipy.ndimage import median_filter
from skimage.exposure import adjust_gamma
from skimage.morphology import remove_small_holes

gamma_adjust = 0.65

vimba_scale_factor = 0.4975
cropsize = 1000


def centercrop(image, cropsize):
    imagesize = image.shape
    xmin = int(0.5 * (imagesize[1] - cropsize))
    xmax = int(0.5 * (imagesize[1] + cropsize))
    ymin = int(0.5 * (imagesize[0] - cropsize))
    ymax = int(0.5 * (imagesize[0] + cropsize))
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image

def findblob(image, sigma):
    image_response = np.zeros(image.shape, dtype=float)
    gaussian_filter(image, sigma, order=2, output=image_response)
    peakloc = peak_local_max(image_response * -1, num_peaks=1)
    peakx = peakloc[0, 1]
    peaky = peakloc[0, 0]
    xa = [peakx - 1, peakx, peakx + 1]
    ya = [image_response[peaky, peakx - 1], image_response[peaky, peakx], image_response[peaky, peakx + 1]]
    xb = [peaky - 1, peaky, peaky + 1]
    yb = [image_response[peaky - 1, peakx], image_response[peaky, peakx], image_response[peaky + 1, peakx]]
    pa = np.polyfit(xa, ya, 2)
    pb = np.polyfit(xb, yb, 2)
    ainterp = -pa[1] / (2 * pa[0])
    binterp = -pb[1] / (2 * pb[0])
    return [ainterp, binterp]


image_stack = imageio.volread('C:/Users/fr293/Qsync/S3_S_0A5' + '.tiff')
image = centercrop(image_stack[133, :, :], cropsize)
image = median_filter(image, size=2)
image = adjust_gamma(image, gamma=gamma_adjust)

radius = int(20/vimba_scale_factor)
selem = disk(radius)

# local_otsu = rank.otsu(img, selem)
# #
# # image = img >= local_otsu

#image = threshold_local(img, 81, offset=10)

thresh = threshold_sauvola(image, window_size=99)

image_thresh = image > thresh

remove_small_holes(image_thresh, area_threshold=2000, in_place=True)

fig, axes = plt.subplots()

axes.imshow(image_thresh, cmap=plt.cm.gray)

[ainterp, binterp] = findblob(image, 40)

circ = Circle([ainterp, binterp], 40, color='red', linewidth=2, fill=False)

axes.add_patch(circ)


plt.show()

