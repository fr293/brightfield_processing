# Fergus Riche 20/09/2019
# This programme uses a Gaussian Process to model the magnetic forces in 3d
# 6 configurations are taken into consideration:
# Current configurations in the old system are 1. UP; 2. LEFT; 3. RIGHT; 4. DOWN; 5, Z-down; 6. Z-up
# Current configurations in the new system are 1. West; 2. South; 3. North; 4. East; 5, Down; 6. Up
# This version also takes into account current magnitude
# 9 magnitudes are available:
# High Range
# 1. 0A5; 2. 1A0; 3. 1A5; 4. 2A0, 5. 2A5
# Low Range
# 6. 0A1; 7. 0A2; 8. 0A3; 9. 0A4


# Libraries
import os
import numpy as np
from sklearn import gaussian_process
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=14)


maxlength = 20000
maxlength_sparse = 1000


# function to load in data
def training_data(filename):
    name = str(filename)

# load in the training data as an np array
    x1 = np.genfromtxt(filename, dtype=np.float, skip_header=1, delimiter=',',
                       encoding=None, usecols=(0, 1, 2, 3, 4, 5))
# randomly subsample training data if the set is too large
    if x1.shape[0] > maxlength_sparse:
        x1 = x1[np.random.choice(x1.shape[0], maxlength_sparse, replace=False), :]

    locations = x1[:, 0:3]
    force_x = x1[:, 3]
    force_y = x1[:, 4]
    force_z = x1[:, 5]

#return a list of arrays (locations, forcex, forcey, forcez)
    return [locations, force_x, force_y, force_z]


# function to train gaussian process
def gp_model(path, filename, no):
    # Note: The unit for x,y,z is um (1e-6m) and the unit for Fx, Fy, Fz is nN (1e-9N)
    cc = str(no)
    path_str = str(path)
    name = str(filename)

    data_list = training_data(os.path.join(path_str, name))

    kernel_x = gaussian_process.kernels.RBF(length_scale=200, length_scale_bounds=(100, 500.0)) + \
               gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    kernel_y = gaussian_process.kernels.RBF(length_scale=200, length_scale_bounds=(100, 500.0)) + \
               gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    kernel_z = gaussian_process.kernels.RBF(length_scale=200, length_scale_bounds=(100.0, 500.0)) + \
               gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    #fit and subtract linear model
    linx = LinearRegression().fit(data_list[0], data_list[1])
    liny = LinearRegression().fit(data_list[0], data_list[2])
    linz = LinearRegression().fit(data_list[0], data_list[3])


    x_predict = linx.predict(data_list[0])
    y_predict = liny.predict(data_list[0])
    z_predict = linz.predict(data_list[0])


    data_adjusted_x = data_list[1] - x_predict
    data_adjusted_y = data_list[2] - y_predict
    data_adjusted_z = data_list[3] - z_predict

    print 'fitting x'
    start = time.time()
    gpx = gaussian_process.GaussianProcessRegressor(kernel=kernel_x, alpha=0.001).\
        fit(data_list[0], data_adjusted_x)
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('x trained width:')
    print (np.exp(gpx.kernel_.theta))
    print 'fitting y'
    start = time.time()
    gpy = gaussian_process.GaussianProcessRegressor(kernel=kernel_y, alpha=0.001).\
        fit(data_list[0], data_adjusted_y)
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('y trained width:')
    print (np.exp(gpy.kernel_.theta))
    print 'fitting z'
    start = time.time()
    gpz = gaussian_process.GaussianProcessRegressor(kernel=kernel_z, alpha=0.001).\
        fit(data_list[0], data_adjusted_z)
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('z trained width:')
    print (np.exp(gpz.kernel_.theta))
    print 'regression finished, saving models'

    # save the model into pkl files
    joblib.dump(gpx, path + '/gp_train_data_sparse_white/config' + cc + 'x.pkl')
    joblib.dump(gpy, path + '/gp_train_data_sparse_white/config' + cc + 'y.pkl')
    joblib.dump(gpz, path + '/gp_train_data_sparse_white/config' + cc + 'z.pkl')
    joblib.dump(linx, path + '/gp_train_data_sparse_white/config' + cc + 'x_lin.pkl')
    joblib.dump(liny, path + '/gp_train_data_sparse_white/config' + cc + 'y_lin.pkl')
    joblib.dump(linz, path + '/gp_train_data_sparse_white/config' + cc + 'z_lin.pkl')

    print("Regression Done")


# function to predict value
def prediction(location_array, gpx, gpy, gpz, linx, liny, linz):

    mx = np.array(gpx.predict(location_array, return_std=True), ndmin=2).T
    my = np.array(gpy.predict(location_array, return_std=True), ndmin=2).T
    mz = np.array(gpz.predict(location_array, return_std=True), ndmin=2).T

    linear_x = linx.predict(location_array).T
    linear_y = liny.predict(location_array).T
    linear_z = linz.predict(location_array).T

    magnitudes = np.column_stack((mx[:, 0] + linear_x, my[:, 0] + linear_y, mz[:, 0]) + linear_z)
    deviations = np.column_stack((mx[:, 1], my[:, 1], mz[:, 1]))

    return [magnitudes, deviations]

def lin_prediction(location_array, gpx, gpy, gpz, linx, liny, linz):

    mx = np.array(gpx.predict(location_array, return_std=True), ndmin=2).T
    my = np.array(gpy.predict(location_array, return_std=True), ndmin=2).T
    mz = np.array(gpz.predict(location_array, return_std=True), ndmin=2).T

    linear_x = linx.predict(location_array).T
    linear_y = liny.predict(location_array).T
    linear_z = linz.predict(location_array).T

    magnitudes = np.column_stack((linear_x, linear_y, linear_z))
    deviations = np.column_stack((mx[:, 0], my[:, 0], mz[:, 0]))

    return [magnitudes, deviations]

def sweep_load(ca, cc):
    # ca is the current magnitude index, going from 1-9
    # cc is the direction index, going from 1-4

    print('loading force data...')
    print('loading: config ' + str(cc) + ', amplitude ' + str(ca))
    predictor_array_x = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'x.pkl')
    predictor_array_y = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'y.pkl')
    predictor_array_z = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'z.pkl')
    lin_model_x = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'x_lin.pkl')
    lin_model_y = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'y_lin.pkl')
    lin_model_z = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data_sparse_white/config' + str(cc) + 'z_lin.pkl')

    return predictor_array_x, predictor_array_y, predictor_array_z, lin_model_x, lin_model_y, lin_model_z

def plot_map(ca):
    # ca is the current magnitude index, going from 1-9
    depth = [-100, 0, 100]
    for d in depth:
        directions = [1, 2, 3, 4]
        mapping = [3, 1, 0, 2]
        amplitude_titles = ['0.5A', '1.0A', '1.5A', '2.0A', '2.5A', '0.1A', '0.2A', '0.3A', '0.4A']

        # fig = plt.figure()
        # ax = plt.subplot2grid((2, 2), (0, 0))
        # ax2 = plt.subplot2grid((2, 2), (0, 1))
        # ax3 = plt.subplot2grid((2, 2), (1, 0))
        # ax4 = plt.subplot2grid((2, 2), (1, 1))
        # axes = [ax4, ax2, ax, ax3]
        fig1, axs1 = plt.subplots(2, 2)
        fig2, axs2 = plt.subplots(2, 2)
        axes1 = axs1.flat
        axes2 = axs2.flat

        limit = 400
        step = 20

        titles = ['West', 'South', 'North', 'East']
        fig1.suptitle('Force CoV Maps for Current Magnitude ' + amplitude_titles[ca-1])
        fig2.suptitle('Force Magnitude Maps for Current Magnitude ' + amplitude_titles[ca - 1])

        for direction in directions:
            predictor_array_x, predictor_array_y, predictor_array_z, lin_model_x, lin_model_y, lin_model_z\
                = sweep_load(ca, direction)
            map = np.mgrid[-limit:limit+step:step, -limit:limit+step:step, d:d + 1]
            map = map.reshape(3, -1).T
            magnitudes, deviations = lin_prediction(map, predictor_array_x, predictor_array_y, predictor_array_z,
                                                lin_model_x, lin_model_y, lin_model_z)

            # coefficient of variation plots

            covs = np.abs(deviations/magnitudes)
            covs[covs >= 1] = 1

            cov_x = covs[:, 0].reshape(41, 41, order='F')
            cov_y = covs[:, 1].reshape(41, 41, order='F')
            cov_z = covs[:, 2].reshape(41, 41, order='F')
            cov_image = np.stack((cov_x, cov_y, cov_z), 2)

            index = mapping[direction - 1]
            axes1[index].set_title(titles[direction-1])
            axes1[index].imshow(cov_image)
            axes1[index].axis('off')
            axes1[index].set_aspect('equal')

            scalebar = AnchoredSizeBar(axes1[index].transData,
                                       10, '100 um', 'lower right',
                                       pad=0.1,
                                       color='grey',
                                       frameon=False,
                                       size_vertical=1,
                                       fontproperties=fontprops)

            axes1[index].add_artist(scalebar)

            fig1.savefig('D:\\sweep_data_new\\' + str(ca) + '\\variation_plot_' + str(d) + '_micron.svg', format='svg')
            plt.close(fig1)

            # magnitude plots

            mags = np.abs(magnitudes)
            max_force = np.amax(mags)
            mags = mags/max_force

            [central_force, devs] = lin_prediction([[0, 0, 0]], predictor_array_x, predictor_array_y, predictor_array_z,
                           lin_model_x, lin_model_y, lin_model_z)

            central_force_x = central_force[0, 0]
            central_force_y = central_force[0, 1]
            central_mag = np.sqrt((central_force_x**2)+(central_force_y**2))
            central_angle = np.degrees(np.arctan(central_force_y/central_force_x))

            mag_x = mags[:, 0].reshape(41, 41, order='F')
            mag_y = mags[:, 1].reshape(41, 41, order='F')
            mag_z = mags[:, 2].reshape(41, 41, order='F')
            mag_image = np.stack((mag_x, mag_y, mag_z), 2)

            axes2[index].set_title(titles[direction-1] + ': $F_{central}$ = ' + format(central_mag, '.2f') + 'nN' +
                                   ' $\measuredangle$ ' + format(central_angle, '.2f') + '$^\circ$', fontsize=9)
            axes2[index].imshow(mag_image)
            axes2[index].axis('off')
            axes2[index].set_aspect('equal')

            scalebar = AnchoredSizeBar(axes2[index].transData,
                                       10, '200 um', 'lower right',
                                       pad=0.1,
                                       color='grey',
                                       frameon=False,
                                       size_vertical=1,
                                       fontproperties=fontprops)

            axes2[index].add_artist(scalebar)

            fig2.savefig('D:\\sweep_data_new\\' + str(ca) + '\\magnitude_plot_' + str(d) + '_micron.svg', format='svg')
            plt.close(fig2)




