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


maxlength = 20000


# function to load in data
def training_data(filename):
    name = str(filename)

# load in the training data as an np array
    x1 = np.genfromtxt(filename, dtype=np.float, skip_header=1, delimiter=',',
                       encoding=None, usecols=(0, 1, 2, 3, 4, 5))
# randomly subsample training data if the set is too large
    if x1.shape[0] > maxlength:
        x1 = x1[np.random.choice(x1.shape[0], maxlength, replace=False), :]

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

    kernel_x_rbf = gaussian_process.kernels.RBF(length_scale=42.0, length_scale_bounds=(40.0, 500.0))
    kernel_y_rbf = gaussian_process.kernels.RBF(length_scale=42.0, length_scale_bounds=(40.0, 500.0))
    kernel_z_rbf = gaussian_process.kernels.RBF(length_scale=42.0, length_scale_bounds=(40.0, 500.0))

    print 'fitting x'
    start = time.time()
    gpx = gaussian_process.GaussianProcessRegressor(kernel=kernel_x, alpha=0.1, normalize_y=True).\
        fit(data_list[0], data_list[1])
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('x trained width:')
    print (np.exp(kernel_x.theta))
    print 'fitting y'
    start = time.time()
    gpy = gaussian_process.GaussianProcessRegressor(kernel=kernel_y, alpha=0.1, normalize_y=True).\
        fit(data_list[0], data_list[2])
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('y trained width:')
    print (np.exp(kernel_y.theta))
    print 'fitting z'
    start = time.time()
    gpz = gaussian_process.GaussianProcessRegressor(kernel=kernel_z, alpha=0.1, normalize_y=True).\
        fit(data_list[0], data_list[3])
    end = time.time()
    clock = end - start
    print ('elapsed time: ' + str(clock))
    print ('z trained width:')
    print (np.exp(kernel_z.theta))
    print 'regression finished, saving models'

    # save the model into pkl files
    joblib.dump(gpx, path + '/gp_train_data/config' + cc + 'x.pkl')
    joblib.dump(gpy, path + '/gp_train_data/config' + cc + 'y.pkl')
    joblib.dump(gpz, path + '/gp_train_data/config' + cc + 'z.pkl')

    print("Regression Done")


# function to predict value
def prediction(location_array, gpx, gpy, gpz):

    mx = np.array(gpx.predict(location_array, return_std=True), ndmin=2).T
    my = np.array(gpy.predict(location_array, return_std=True), ndmin=2).T
    mz = np.array(gpz.predict(location_array, return_std=True), ndmin=2).T

    magnitudes = np.column_stack((mx[:, 0], my[:, 0], mz[:, 0]))
    deviations = np.column_stack((mx[:, 1], my[:, 1], mz[:, 1]))

    return [magnitudes, deviations]


def sweep_load(ca, cc):
    # ca is the current magnitude index, going from 1-9
    # cc is the direction index, going from 1-4

    print('loading force data...')
    print('loading: config ' + str(cc) + ', amplitude ' + str(ca))
    predictor_array_x = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data/config' + str(cc) + 'x.pkl')
    predictor_array_y = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data/config' + str(cc) + 'y.pkl')
    predictor_array_z = joblib.load('D://sweep_data_new/' + str(ca) + '/gp_train_data/config' + str(cc) + 'z.pkl')

    return predictor_array_x, predictor_array_y, predictor_array_z

def plot_map(ca):
    # ca is the current magnitude index, going from 1-9

    map = np.mgrid[-400:420:20, -400:420:20, 0:1]
    map = map.reshape(3, -1).T

    directions = [3, 2, 4, 1]

    for direction in directions:
        predictor_array_x, predictor_array_y, predictor_array_z = sweep_load(ca, direction)
        magnitudes, deviations = prediction(map, predictor_array_x, predictor_array_y, predictor_array_z)

        magnitudes = magnitudes.reshape(41,41, order='F')
        mag_image = np.rot90(magnitudes)


