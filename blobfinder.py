import imageio
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_sauvola
from skimage.exposure import adjust_gamma
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import resize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from tqdm import tqdm
import GP_force_predictor as forcePredict
import re
matplotlib.use('TkAgg')

cropsize = 1000
cropsize_tracking = 300
blobsize = 40
gamma_adjust = 0.7
resize_factor = 2
threshold_size = 94
hole_threshold_area = 2000
object_threshold_area = 4000
movement_threshold = 2
vimba_scale_factor = 0.4975
offset_x = 1294
offset_y = 970
n_glass = 1.474
n_water = 1.333


def round_odd(number):
    return int(2*np.floor(number/2)+1)


def centercrop(image, crop_dimension):
    imagesize = image.shape
    xmin = int(0.5 * (imagesize[1] - crop_dimension))
    xmax = int(0.5 * (imagesize[1] + crop_dimension))
    ymin = int(0.5 * (imagesize[0] - crop_dimension))
    ymax = int(0.5 * (imagesize[0] + crop_dimension))
    corner = np.array([xmin, ymin])
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image, corner


def trackingcrop(image, tracking_radius, cropcenter):
    # danger ahead: be careful around coordinate ordering
    # image.shape returns the format [y,x], and cropcenter is in [x,y]
    xmin = int(cropcenter[0] - 0.5 * tracking_radius)
    xmax = int(cropcenter[0] + 0.5 * tracking_radius)
    ymin = int(cropcenter[1] - 0.5 * tracking_radius)
    ymax = int(cropcenter[1] + 0.5 * tracking_radius)
    cropped_image = image[ymin:ymax, xmin:xmax]
    corner = np.array([xmin, ymin])
    return cropped_image, corner


def findblob(image, filter_size):
    # takes an image and looks for a blob of the specified size
    # returns the coordinates of the blob in pixels, centred on the image
    image_response = gaussian_filter(image, filter_size, order=2, output=float)
    peakloc = peak_local_max(image_response * -1, num_peaks=1)
    try:
        peakx = peakloc[0, 1]
        peaky = peakloc[0, 0]
    except IndexError:
        peakx = 0
        peaky = 0
    xa = [peakx - 1, peakx, peakx + 1]
    ya = [image_response[peaky, peakx - 1], image_response[peaky, peakx], image_response[peaky, peakx + 1]]
    xb = [peaky - 1, peaky, peaky + 1]
    yb = [image_response[peaky - 1, peakx], image_response[peaky, peakx], image_response[peaky + 1, peakx]]
    pa = np.polyfit(xa, ya, 2)
    pb = np.polyfit(xb, yb, 2)
    x_shift = -pa[1] / (2 * pa[0])
    y_shift = -pb[1] / (2 * pb[0])
    return x_shift, y_shift


def computestrain(position_stack, force_stack):
    stackshape = position_stack.shape
    length = stackshape[0]

    dotted_distance_stack = np.zeros([length, 1])
    alignment_ratio = np.zeros([length, 1])
    force_magnitude_stack = np.zeros([length, 1])
    # make starting position the datum
    position_stack = position_stack - position_stack[0, :]
    position_stack_incremental = np.concatenate((np.zeros([1, 3]), np.diff(position_stack, axis=0)))

    for k in range(length):
        f_mag = np.linalg.norm(force_stack[k, :2])
        if f_mag != 0:
            force_magnitude_stack[k] = f_mag
            unit_force = force_stack[k, :2] / force_magnitude_stack[k]
            dotted_distance_stack[k] = np.dot(position_stack_incremental[k, :2], unit_force[:2])
            c = np.cross(position_stack_incremental[k, :2].T, unit_force[:2].T)
            s = np.sign(c)
            if np.abs(dotted_distance_stack[k]) == 0:
                alignment_ratio[k] = 0
            else:
                alignment_ratio[k] = s*(np.sqrt(np.linalg.norm(position_stack_incremental[k, :2])**2
                                                - dotted_distance_stack[k]**2))/np.abs(dotted_distance_stack[k])

    dotted_distance_stack = np.cumsum(dotted_distance_stack, axis=0)

    return dotted_distance_stack, alignment_ratio, force_magnitude_stack


def locator(image):
    image = resize(image, [image.shape[0] / resize_factor, image.shape[1] / resize_factor], anti_aliasing=True,
                   mode='reflect')
    image = median_filter(image, size=2)
    image = adjust_gamma(image, gamma=gamma_adjust)
    thresh = threshold_sauvola(image, window_size=round_odd(threshold_size / resize_factor))
    image_thresh = image > thresh
    remove_small_holes(image_thresh, area_threshold=np.floor(hole_threshold_area / resize_factor ** 2), in_place=True)
    remove_small_objects(image_thresh, min_size=np.floor(object_threshold_area / resize_factor ** 2), in_place=True)
    x, y = findblob(image_thresh, blobsize / resize_factor)
    x = x * resize_factor
    y = y * resize_factor
    return x, y, image_thresh


# this function finds the last value that's smaller than the nominal on or off time
def index_finder(time_vector, critical_time):
    difference = time_vector - critical_time
    difference[difference > 0] = np.nan
    index = np.nanargmin(np.abs(difference))
    return index


def force_switch_indices_semantic(filename, time_vector):
    # all filenames end with a two character duration value
    # extract this value and use it to interpret the indices
    # hand-tuned start and end index values are listed in comments for each condition
    duration_value = filename[-2:]

    if duration_value == '10':
        on_time = 10
        off_time = 20
        # start_index = 19
        # end_index = 35

    elif duration_value == '30':
        on_time = 10
        off_time = 40
        # start_index = 19
        # end_index = 71

    elif duration_value == '90':
        on_time = 10
        off_time = 100
        # start_index = 19
        # end_index = 188

    else:
        print('error: filename does not end with duration code')
        raise UserWarning

    start_index = index_finder(time_vector, on_time)
    end_index = index_finder(time_vector, off_time)

    return start_index, end_index


def findblobstack(image_filename, image_filepath, output_filepath, ca, cc):
    try:
        image_stack = imageio.volread(image_filepath + image_filename + '.tif')
    except OSError:
        image_stack = imageio.volread(image_filepath + image_filename + '.tiff')

    try:
        time_stack = np.array(np.genfromtxt(image_filepath + image_filename + '_time.csv', dtype=float, delimiter=','),
                              ndmin=2).T
    except IOError:
        image_filename = re.sub('_r', '', image_filename)
        time_stack = np.array(np.genfromtxt(image_filepath + image_filename + '_time.csv', dtype=float, delimiter=','),
                              ndmin=2).T

    try:
        z_values = np.genfromtxt(image_filepath + 'Z_information.csv', dtype=float, delimiter=',')
    except IOError:
        image_filename = re.sub('_r', '', image_filename)
        z_values = np.genfromtxt(image_filepath + 'Z_information.csv', dtype=float, delimiter=',')

    stackshape = image_stack.shape
    length = stackshape[0]
    height_offset = stackshape[1] * 0.5
    width_offset = stackshape[2] * 0.5
    position_stack = np.zeros([length, 3])
    force_mask = np.zeros([length, 1])
    # make starting time the datum
    time_stack = time_stack - time_stack[0]
    z_measured = (z_values[0]-z_values[1])*1000
    z_actual = 500 - (((1 - (n_water/n_glass))*200) + n_water*z_measured)

    # Initial blob finder The values in the morphological filters are dependent on the image size, which is resized
    # here to 0.5 its original dimension
    image, corner = centercrop(image_stack[0, :, :], cropsize)
    x, y, image_thresh = locator(image)
    position_stack[0, 0:2] = np.array([x+corner[0], y+corner[1]])

    # Use previous blob location to inform where to look next
    for i in tqdm(range(1, length)):
        image, corner = trackingcrop(image_stack[i, :, :], cropsize_tracking, position_stack[i - 1, :2])
        x, y, image_thresh = locator(image)
        position_stack[i, 0:2] = np.array([x+corner[0], y+corner[1]])

    # add in the z coordinate, and scale the plane coordinates by the hardware factor
    position_stack[:, 2] = z_actual
    position_stack[:, :2] = position_stack[:, :2] - [width_offset, height_offset]
    position_stack[:, :2] = position_stack[:, :2] * vimba_scale_factor

    lin_model_x, lin_model_y, lin_model_z = forcePredict.sweep_load_lin(ca, cc)

    force_stack = forcePredict.lin_prediction(position_stack, lin_model_x, lin_model_y, lin_model_z)

    start_index, end_index = force_switch_indices_semantic(image_filename, time_stack)

    force_mask[start_index:end_index] = 1

    eigendisplacement, alignment_ratio, eigenforce = computestrain(position_stack, force_stack)

    eigenforce = np.multiply(eigenforce, force_mask)

    fig = plt.figure()
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax4 = ax3.twinx()
    fig.suptitle(image_filename)

    plotting_image, corner = centercrop(image_stack[0, :, :], cropsize)

    # this axis plots the raw image
    ax.cla()
    ax.set_title('Raw Image')
    ax.imshow(plotting_image, interpolation='nearest')
    ax.axis('off')
    ax.set_aspect('equal')

    # this block plots the thresholded image, along with the bead location and force vector
    # note that the scaling factor has to be removed for the annotations
    ax2.cla()
    ax2.set_title('Annotated Image')
    ax2.imshow(plotting_image, interpolation='nearest')
    ax2.axis('off')
    ax2.set_aspect('equal')
    (a, b, c) = position_stack[length-1, :]
    im_x = int((a / vimba_scale_factor) + width_offset - corner[0])
    im_y = int((b / vimba_scale_factor) + height_offset - corner[1])
    circ = Circle((im_x, im_y), blobsize/2, color='red', linewidth=2, fill=False)
    arrow = Arrow(im_x, im_y, 100 * force_stack[length-1, 0], 100 * force_stack[length-1, 1], width=30, color='black')
    ax2.add_patch(circ)
    ax2.add_patch(arrow)
    ax2.scatter(x=(position_stack[0:length-1, 0] / vimba_scale_factor) + width_offset - corner[0],
                y=position_stack[0:length-1, 1] / vimba_scale_factor + height_offset - corner[1], c='orange', s=10)

    # this axis plots the strain in the direction of the force, the strain normal to the force
    # and the force magnitude
    ax3.cla()
    ax3.set_xlim([0, time_stack[length - 1]])
    ax3.set_ylim([-1, 1.1 * np.amax(eigendisplacement[0:length-1])])
    ax3.plot(time_stack[0:length-1], eigendisplacement[0:length-1], color='blue', linewidth=2, label='creep distance')

    ax3.set_ylabel('Bead Displacement/um')
    ax3.set_xlabel('Time/s')
    ax3.legend(loc='upper right', fontsize='medium', framealpha=0.5)

    ax4.cla()
    ax4.set_title('Bead Displacement and Force Magnitude')
    ax4.set_xlabel('Time/s')
    ax4.set_ylabel('Force/nN')
    ax4.plot(time_stack[0:length - 1], eigenforce[0:length - 1], color='black', linewidth=2, label='force magnitude')
    ax4.set_ylim(0, 1.1*np.max(eigenforce[0:length - 1]))

    plt.savefig(output_filepath + image_filename + '.jpeg')
    plt.close(fig)

    return [time_stack, position_stack, force_stack, force_mask, eigenforce, eigendisplacement, alignment_ratio]


def thresholding_matrix():
#    image_stack = imageio.volread('D:\sync_folder\experiments_DNA_Brushes\A_series_2uM\\brightfield\A1\A1_E_0A1_10.tiff')
    image_stack = imageio.volread('D:\\sync_folder\\experiments_DNA_Brushes'
                                  '\\B_series_0uM2\\brightfield\\B1\\B1_E_0A2_30.tiff')

    crop_region = 1900

    gamma_adjust_values = [0.5, 0.6, 0.7, 0.8]
    window_sizes = [11, 23, 47, 95]

    fig, ax = plt.subplots(4, 4)

    for i in range(4):
        for j in range(4):
            image = centercrop(image_stack[1, :, :], crop_region)
            image = resize(image, [crop_region / resize_factor, crop_region / resize_factor], anti_aliasing=True,
                           mode='reflect')
            image = median_filter(image, size=2)
            image = adjust_gamma(image, gamma=gamma_adjust_values[i])
            thresh = threshold_sauvola(image, window_size=window_sizes[j])
            image_thresh = image > thresh
            remove_small_holes(image_thresh, area_threshold=np.floor(hole_threshold_area / resize_factor ** 2),
                               in_place=True)
            remove_small_objects(image_thresh, min_size=np.floor(object_threshold_area / resize_factor ** 2),
                                 in_place=True)
            x, y = findblob(image_thresh, 40 / resize_factor)
            circ = Circle((x+475, y+475), 40 / 2, color='red', linewidth=2, fill=False)

            ax[i, j].imshow(image_thresh)
            ax[i, j].add_patch(circ)
