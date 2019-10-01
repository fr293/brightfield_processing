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
from timeit import default_timer as timer


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

    kernel = gaussian_process.kernels.RBF(length_scale=1, length_scale_bounds=(1e-2, 1000.0))
    print 'fitting x'
    start = timer()
    gpx = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=0.1).fit(data_list[0], data_list[1])
    end = timer()
    clock = end - start
    print ('elapsed time: ' + clock)
    print 'fitting y'
    start = timer()
    gpy = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=0.1).fit(data_list[0], data_list[2])
    end = timer()
    clock = end - start
    print ('elapsed time: ' + clock)
    print 'fitting z'
    start = timer()
    gpz = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=0.1).fit(data_list[0], data_list[3])
    end = timer()
    clock = end - start
    print ('elapsed time: ' + clock)
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
    predictor_array_x = joblib.load('D://sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'x.pkl')
    predictor_array_y = joblib.load('D://sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'y.pkl')
    predictor_array_z = joblib.load('D://sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'z.pkl')

    return predictor_array_x, predictor_array_y, predictor_array_z
