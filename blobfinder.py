import imageio
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_sauvola
from skimage.exposure import adjust_gamma
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from tqdm import tqdm
import GP_force_predictor as f
import re

movement_threshold = 2
vimba_scale_factor = 0.4975
offset_x = 1294
offset_y = 970
n_glass = 1.474
n_water = 1.333


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
    x_shift_scale = (ainterp - offset_x) * vimba_scale_factor
    y_shift_scale = (binterp - offset_y) * vimba_scale_factor
    return [x_shift_scale, y_shift_scale]


def computestrain(position_stack, force_stack):
    stackshape = position_stack.shape
    length = stackshape[0]

    dotted_distance_stack = np.zeros([length, 1])
    orthogonal_distance_stack = np.zeros([length, 1])
    force_magnitude_stack = np.zeros([length, 1])
    # make starting position the datum
    position_stack = position_stack - position_stack[0, :]

    for k in range(length):
        if sum(force_stack[k, :]) != 0:
            force_magnitude_stack[k] = np.linalg.norm(force_stack[k, :])
            unit_force = force_stack[k, :] / force_magnitude_stack[k]
            dotted_distance_stack[k] = np.dot(position_stack[k, :], unit_force)
            handedness = np.cross(position_stack[k, :], unit_force)
            orthogonal_distance_stack[k] = np.sign(handedness[2]) * np.sqrt(
                np.linalg.norm(position_stack[k, :]) - np.linalg.norm(dotted_distance_stack[k]))
            orthogonal_distance_stack[k] = 0

    return dotted_distance_stack, orthogonal_distance_stack, force_magnitude_stack


def findblobstack(image_filename, image_filepath, output_filepath, ca, cc, cropsize, sigma, gamma_adjust):
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

    force_on = False
    stackshape = image_stack.shape
    length = stackshape[0]
    position_stack = np.zeros([length, 3])
    force_stack = np.zeros([length, 3])
    force_mask = np.ones([length, 1])
    # make starting time the datum
    time_stack = time_stack - time_stack[0]
    z_measured = (z_values[0]-z_values[1])*1000
    z_actual = 500 - (((1 - (n_water/n_glass))*200) + n_water*z_measured)

    dummy_direction = np.array([[-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 0, 0]])

    plt.ion()
    fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 2)
    # ax2 = fig.add_subplot(1, 2, 1)
    ax = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax4 = ax3.twinx()
    fig.suptitle(image_filename)

    # print('processing images and calculating forces...')

    for i in tqdm(range(0, length)):
        image = centercrop(image_stack[i, :, :], cropsize)
        image = median_filter(image, size=2)
        image = adjust_gamma(image, gamma=gamma_adjust)

        thresh = threshold_sauvola(image, window_size=45)
        image_thresh = image > thresh
        remove_small_holes(image_thresh, area_threshold=2000, in_place=True)
        remove_small_objects(image_thresh, min_size=2000, in_place=True)
        x, y = findblob(image_thresh, sigma)
        position_stack[i, 0:2] = [x, y]
        force_stack[i, :] = dummy_direction[cc-1, :]
        dotted_distance_stack, orthogonal_distance_stack, force_magnitude_stack = computestrain(position_stack,
                                                                                                force_stack)

        diff = dotted_distance_stack[i] - dotted_distance_stack[i - 1]
        if float(diff) > movement_threshold:
            force_on = True

        elif float(diff) < -movement_threshold:
            force_on = False

        if not force_on:
            force_mask[i] = 0

        # this axis plots the raw image
        ax.cla()
        ax.set_title('Raw Image')
        ax.imshow(image, interpolation='nearest')
        ax.axis('off')
        ax.set_aspect('equal')

        # this block plots the thresholded image, along with the bead location and force vector
        ax2.cla()
        ax2.set_title('Processed Image')
        ax2.imshow(image_thresh, interpolation='nearest')
        ax2.axis('off')
        ax2.set_aspect('equal')
        (a, b, c) = position_stack[i, :]
        im_x = int((a/vimba_scale_factor) + offset_x)
        im_y = int((b/vimba_scale_factor) + offset_y)
        circ = Circle((im_x, im_y), sigma, color='red', linewidth=2, fill=False)
        arrow = Arrow(im_x, im_y, 100 * force_stack[i, 0], 100 * force_stack[i, 1], width=30, color='black')
        ax2.add_patch(circ)
        if force_on:
            ax2.add_patch(arrow)
        ax2.scatter(x=(position_stack[0:i, 0]/vimba_scale_factor) + offset_x,
                    y=position_stack[0:i, 1]/vimba_scale_factor + offset_y, c='g', s=10)

        # this axis plots the strain in the direction of the force, the strain normal to the force
        # and the force magnitude
        ax3.cla()
        ax3.set_xlim([0, time_stack[length - 1]])
        if any(dotted_distance_stack[0:i] > 18):
            ax3.set_ylim([-5, 1.1*np.amax(dotted_distance_stack[0:i])])
        else:
            ax3.set_ylim([-5, 20])

        distance_along, = ax3.plot(time_stack[0:i], dotted_distance_stack[0:i], color='blue',
                                   linewidth=2, label='distance along force vector')
        distance_residual, = ax3.plot(time_stack[0:i], orthogonal_distance_stack[0:i],
                                      color='orange', linewidth=2, label='distance residual')

        ax3.set_ylabel('Bead Displacement/um')
        ax3.set_xlabel('Time/s')

        ax4.cla()
        ax4.set_title('Bead Displacement and Force Magnitude')
        #ax4.set_xlim([0, time_stack[length - 1]])
        #ax4.set_ylim([0, 10])
        #force_magnitude, = ax4.plot(time_stack[0:i], force_magnitude_stack[0:i], color='red', linewidth=2)
        ax4.set_xlabel('Time/s')
        #ax4.set_ylabel('Force Magnitude/nN')
        # ax4.legend([distance_along, distance_residual, force_magnitude],
        #            ['distance along', 'distance residual', 'force magnitude'])
        ax4.legend([distance_along, distance_residual,],
                   ['distance along', 'distance residual'])

        plt.draw()
        plt.pause(0.01)
    plt.savefig(output_filepath + image_filename + '.svg')
    plt.savefig(output_filepath + image_filename + '.jpeg')
    plt.close(fig)

    predictor_array_x, predictor_array_y, predictor_array_z, lin_model_x, lin_model_y, lin_model_z\
        = f.sweep_load(ca, cc)

    position_stack[:, 2] = z_actual

    force_data = f.prediction(position_stack, predictor_array_x, predictor_array_y, predictor_array_z,
                              lin_model_x, lin_model_y, lin_model_z)

    dotted_distance_stack, orthogonal_distance_stack, force_magnitude_stack = computestrain(position_stack,
                                                                                            force_data[0])
    force_magnitude_stack = np.multiply(force_magnitude_stack, force_mask)

    return [time_stack, position_stack, dotted_distance_stack, force_data[0], force_data[1], force_mask]

