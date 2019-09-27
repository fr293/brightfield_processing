# Qian Cheng 11/11/2015
# Fergus Riche 26/03/2019
# This programme uses Gaussian Process to model the magnetic forces in 3d
# N.b. this requires scikit-learn 0.16.0
# 6 configurations are taken into consideration:
# Current configurations in the old system are 1. UP; 2. LEFT; 3. RIGHT; 4. DOWN; 5, Z-down; 6. Z-up
# Current configurations in the new system are 1. West; 2. South; 3. North; 4. East; 5, Down; 6. Up
# This version also takes into account current magnitude
# 9 magnitudes are available:
# the high force range
# 1. 0A5; 2. 1A0; 3. 1A5; 4. 2A0; 5. 2A5
# the low force range
# 6. 0A1; 7. 0A2;  8. 0A3; 9. 0A4

# Libraries
from math import sqrt
from pylab import *
import sys
import csv
from sympy import *
from sklearn import gaussian_process
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import collections
from sklearn.externals import joblib
from tqdm import tqdm


#GP Model
# Extract Data From Folders and return in Objective Functions
def objective_function(filename, no):
    cc = str(no)
    name = str(filename)
    # This function is to create the training data set as the return of the function
    # The inputs are the current configuration and the Cartesian coordinates of the bead
    # The output are the difference between monopole prediction and the real value for x, y, z respectively

    inputdata = []
    traindata = []
    train1 = []
    train2 = []

    # reading data from the files
    csv_in1 = csv.reader(open(name, 'rb'), delimiter=',')
    # csv_in2 = csv.reader(open('train'+cc+'.csv', 'rb'), delimiter = ',')

    # extract data from the csv file
    for row1 in csv_in1:
        inputdata.append(row1)

    # for row2 in csv_in2:
    #	traindata.append(row2)

    # convert the data type to float
    x1 = np.array(inputdata)
    x1 = x1[2:, :6]
    train1 = x1.astype(float)
    train1 = np.array(train1).tolist()

    x2 = x1[:, :3]
    train2 = x2.astype(float)

    train1 = zip(*train1)

    train = np.array(train2).tolist()

    targetx = train1[3]
    targety = train1[4]
    targetz = train1[5]
    # targetf = train1[6]

    # print len(targetx)
    # print(targetx[1:10])

    el = zip(*train1)
    y = collections.Counter(el[0])

    #    with open('togger.csv', 'w') as csvfile:
    #        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #        for elements in targetx:
    #            spamwriter.writerow([elements])

    return train, targetx, targety, targetz  # , targetf


# Gaussian Process Regression Model
def gp_model(path, filename, no):
    cc = str(no)
    path = str(path)
    name = str(filename)

    # Define Gaussian Process (GP)

    gpx = gaussian_process.GaussianProcess(corr='squared_exponential', optimizer='fmin_cobyla', theta0=1e-2,
                                           storage_mode='light', thetaL=1e-2, thetaU=100, nugget=0.1)
    gpy = gaussian_process.GaussianProcess(corr='squared_exponential', optimizer='fmin_cobyla', theta0=1e-2,
                                           storage_mode='light', thetaL=1e-2, thetaU=100, nugget=0.1)
    gpz = gaussian_process.GaussianProcess(corr='squared_exponential', optimizer='fmin_cobyla', theta0=1e-2,
                                           storage_mode='light', thetaL=1e-2, thetaU=100, nugget=0.1)
    # gpf = gaussian_process.GaussianProcess(corr='squared_exponential', optimizer='fmin_cobyla', theta0=1e-2,  storage_mode = 'light', thetaL=1e-2, thetaU=100, nugget=0.1)

    ##Once the data is ready, include the regression model train is the train data targetx/y/z is the target

    data_list = objective_function(os.path.join(path, name), cc)

    # Note: The unit for x,y,z is mm (1e-3m) and the unit for Fx, Fy, Fz is nN (1e-9N)

    # Fit x component
    print('fitting x')
    try:
        gpx.fit(data_list[0], data_list[1])
    except:
        print('Error in fitting. Carrying on...')

    # Fit y component
    print('fitting y')
    try:
        gpy.fit(data_list[0], data_list[2])
    except:
        print('Error in fitting. Carrying on...')

    # Fit z component
    print('fitting z')
    try:
        gpz.fit(data_list[0], data_list[3])
    except:
        print('Error in fitting. Carrying on...')

    ##Fit f componet
    # gpf.fit(data_list[0], data_list[4])

    print "Regression Finished"

    l = gpx.get_params(True)
    m = gpx.get_params(True)
    n = gpx.get_params(True)

    ########Save the model into pkl files########

    joblib.dump(gpx, path + '/gp_train_data/config' + cc + 'x.pkl')
    joblib.dump(gpy, path + '/gp_train_data/config' + cc + 'y.pkl')
    joblib.dump(gpz, path + '/gp_train_data/config' + cc + 'z.pkl')
    # joblib.dump(gpz,'../../Data/Exp4_20151102/GP_model/config'+cc+'f.pkl')

    print("Regression Done")


def sweep_load():
    # ca is the current magnitude index, going from 1-5
    # cc is the direction index, going from 1-4

    predictor_array_x = np.empty((5, 4), dtype=object)
    predictor_array_y = np.empty((5, 4), dtype=object)
    predictor_array_z = np.empty((5, 4), dtype=object)

    #print('loading force data...')
    for ca in range(1, 6):
        for cc in range(1, 5):
            #print('loading: config ' + str(cc) + ', amplitude ' + str(ca))
            predictor_array_x[ca-1, cc-1] = joblib.load('sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'x.pkl')
            predictor_array_y[ca-1, cc-1] = joblib.load('sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'y.pkl')
            predictor_array_z[ca-1, cc-1] = joblib.load('sweep_data/' + str(ca) + '/gp_train_data/config' + str(cc) + 'z.pkl')

    return predictor_array_x, predictor_array_y, predictor_array_z


# Making Predictions
def prediction(ca, cc, x, y, z, predictor_array_x, predictor_array_y, predictor_array_z):

    gp1x = predictor_array_x[ca - 1, cc - 1]
    gp1y = predictor_array_y[ca - 1, cc - 1]
    gp1z = predictor_array_z[ca - 1, cc - 1]

    gpx, sigma1 = gp1x.predict([x, y, z], eval_MSE=True)
    gpy, sigma2 = gp1y.predict([x, y, z], eval_MSE=True)
    gpz, sigma3 = gp1z.predict([x, y, z], eval_MSE=True)

    gpx = float(gpx)
    gpy = float(gpy)
    gpz = float(gpz)
    sigma1 = float(sigma1)
    sigma2 = float(sigma2)
    sigma3 = float(sigma3)

    return gpx, gpy, gpz, sigma1, sigma2, sigma3


# Data Plots
# Plot the training data points
def train_data_plot(no):
    cc = str(no)

    # Save the raw data from the experimental measurement
    rawdata = []

    # reading data from the files
    csv_in1 = csv.reader(open('conf' + cc + '.csv', 'rb'), delimiter=',')

    # extract data from the csv file
    for row1 in csv_in1:
        rawdata.append(row1)

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    x1 = np.array(x1).tolist()

    x1z = zip(*x1)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_zlim(0.2, -0.4)
    # ax.set_ylim(0.3, -0.3)
    # ax.set_xlim(-0.2, 0.4)
    ax.scatter(x1z[0], x1z[1], x1z[2], color='red')

    ax.set_xlabel('X (mm)', fontsize=16)
    ax.set_ylabel('Y (mm)', fontsize=16)
    ax.set_zlabel('Z (mm)', fontsize=16)
    ax.legend()
    matplotlib.pyplot.show()


def makeforcedata(no, z_center):
    cc = str(no)

    #Gaussian Process (GP)
    gpx = joblib.load('config' + cc + 'x.pkl')
    gpy = joblib.load('config' + cc + 'y.pkl')
    gpz = joblib.load('config' + cc + 'z.pkl')

    data = []

    z = z_center
    # y = 0.0

    mod_trans = 0.70710678118  # sqrt(2)/2

    for x in xrange(-25, 25, 1):

        for y in xrange(-25, 25, 1):
            # for z in xrange(-0.2, 0.2, 0.005):

            #this block rotates the x,z coordinates about y if necessary
            # xx = (mod_trans * x + (-1)*mod_trans * z) + 0.100

            # yy = y

            # zz = (mod_trans * x + mod_trans * z) - 0.100

            xx = x

            yy = y

            zz = z

            gp1, sigma1 = gpx.predict([xx, yy, zz], eval_MSE=True)
            gp2, sigma2 = gpy.predict([xx, yy, zz], eval_MSE=True)
            gp3, sigma3 = gpz.predict([xx, yy, zz], eval_MSE=True)

            force = sqrt(gp1 * gp1 + gp2 * gp2 + gp3 * gp3)

            data.append([x, y, z, gp1, gp2, gp3, force, math.atan2(gp2, gp1)])

            print "here"

    f1 = np.array(data)
    f1 = f1.astype(float)

    test1 = np.array(f1).tolist()

    with open('forcedata' + cc + '.csv', "wb") as f:

        writer = csv.writer(f, delimiter='\t')

        writer.writerows(f1)

    # el2 = zip(*data)


def xrange(start, stop, step):
    while start < stop:
        yield start
        start += step
