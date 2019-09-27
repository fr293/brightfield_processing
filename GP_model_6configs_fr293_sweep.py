# Qian Cheng 11/11/2015
# This programme uses Gaussian Process to model the magnetic forces in 3d
# 6 configurations are taken into consideration:
# Current configurations in the old system are 1. UP; 2. LEFT; 3. RIGHT; 4. DOWN; 5, Z-down; 6. Z-up
# Current configurations in the new system are 1. West; 2. South; 3. North; 4. East; 5, Down; 6. Up
# This version also takes into account current magnitude
# 9 magnitudes are available:
# High Range
# 1. 0A5; 2. 1A0; 3. 1A5; 4. 2A0, 5. 2A5
# Low Range
# 6. 0A1; 7. 0A2; 8. 0A3; 9. 0A4


################################## Libraries ########################################
import os
import numpy as np
from numpy import ma

import math
from math import pow
from math import sqrt
from scipy import stats
from scipy.integrate import odeint
from scipy.optimize import leastsq
from pylab import *
import fileinput, sys, csv
from sympy import *
# import scikits.statsmodels.api as sm
from sympy.functions import coth
from sympy.functions import cosh
from sympy.functions import sinh

### import the library of gaussian process regression (GP) ###
from sklearn import svm
from sklearn.svm import SVR
from sklearn import gaussian_process

import random
from IPython.display import display
### Python Plot Library #################
import matplotlib.pyplot as pl
import pylab
from mpl_toolkits.mplot3d import Axes3D
import collections
import pickle
from sklearn.externals import joblib
# from mayavi.mlab import *
from matplotlib import cm


####################################################################################

###############################GP Model#############################################

######### Extract Data From Folders and return in Objective Functions ##############	
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


################## Gaussian Process Regression Model ###############################
def gp_model(path, filename, no):
    cc = str(no)
    path = str(path)
    name = str(filename)

    ####################Define Gaussian Process (GP)######################

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
    #try:
    gpx.fit(data_list[0], data_list[1])
        #except:
        #print('Error in fitting. Carrying on...')

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
    m = gpy.get_params(True)
    n = gpz.get_params(True)

    ########Save the model into pkl files########

    joblib.dump(gpx, path + '/gp_train_data/config' + cc + 'x.pkl')
    joblib.dump(gpy, path + '/gp_train_data/config' + cc + 'y.pkl')
    joblib.dump(gpz, path + '/gp_train_data/config' + cc + 'z.pkl')
    # joblib.dump(gpz,'../../Data/Exp4_20151102/GP_model/config'+cc+'f.pkl')

    print("Regression Done")


################# Making Predictions  ###################
def prediction(current_amplitude, no, x, y, z):
    cc = str(no)
    ca = str(current_amplitude)

    ####################Gaussian Process (GP)######################
    gp1x = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'x.pkl')
    gp1y = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'y.pkl')
    gp1z = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'z.pkl')

    ########The below starts the prediction process ################

    gpx, sigma1 = gp1x.predict([x, y, z], eval_MSE=True)
    gpy, sigma2 = gp1y.predict([x, y, z], eval_MSE=True)
    gpz, sigma3 = gp1z.predict([x, y, z], eval_MSE=True)

    gpx = float(gpx)
    gpy = float(gpy)
    gpz = float(gpz)
    return gpx, gpy, gpz


def prediction_multiple(input_file, output_file, current_amplitude, no):
    cc = str(no)
    ca = str(current_amplitude)

    ####################Reading Data######################
    rawdata = []
    newdata = []

    # reading data from the files
    csv_in1 = csv.reader(open(input_file, 'rb'), delimiter=',')

    # extract data from the csv file
    for row1 in csv_in1:
        rawdata.append(row1)

    # print len(rawdata[0])

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    rawdata = np.array(x1).tolist()

    gp1x = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'x.pkl')
    gp1y = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'y.pkl')
    gp1z = joblib.load('sweep_data/' + ca + '/gp_train_data/config' + cc + 'z.pkl')

    for i in range(len(rawdata)):
        x = float(rawdata[i][0])
        y = float(rawdata[i][1])
        z = float(rawdata[i][2])

    #	print x,y,z

    gpx, sigma1 = gp1x.predict([x, y, z], eval_MSE=True)
    gpy, sigma2 = gp1y.predict([x, y, z], eval_MSE=True)
    gpz, sigma3 = gp1z.predict([x, y, z], eval_MSE=True)

    gpx = float(gpx)
    gpy = float(gpy)
    gpz = float(gpz)

    newdata.append([rawdata[i][0], rawdata[i][1], rawdata[i][2], gpx, gpy, gpz])

    with open(output_file, "wb") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(newdata)


##################### Validation Checking Errors #################################
####Give a configuration number####	
def validation(no):
    cc = str(no)

    testdata1 = []
    testdata2 = []
    errordata = []

    # reading data from the files
    csv_in1 = csv.reader(open('/group/data-d2/Qian/Calibration/Data/Exp4_20151102/Config' + cc + '/valid.csv', 'rb'),
                         delimiter='\t')

    # extract data from the csv file
    for row1 in csv_in1:
        testdata1.append(row1)

    x1 = np.array(testdata1)
    train1 = x1.astype(float)
    test1 = np.array(train1).tolist()

    gpx = joblib.load('../../Data/Exp2_20150902/GP_model/config' + cc + 'x.pkl')
    gpy = joblib.load('../../Data/Exp2_20150902/GP_model/config' + cc + 'y.pkl')
    gpz = joblib.load('../../Data/Exp2_20150902/GP_model/config' + cc + 'z.pkl')

    ###########Error Analysis################
    for i in range(len(test1)):
        ###making predictions
        gp1, sigma1 = gpx.predict([test1[i][0] * 1.e-3, test1[i][1] * 1.e-3, test1[i][2] * 1.e-3], eval_MSE=True)
        gp2, sigma2 = gpy.predict([test1[i][0] * 1.e-3, test1[i][1] * 1.e-3, test1[i][2] * 1.e-3], eval_MSE=True)
        gp3, sigma3 = gpz.predict([test1[i][0] * 1.e-3, test1[i][1] * 1.e-3, test1[i][2] * 1.e-3], eval_MSE=True)

        ###calculating the error

        Ex = abs(test1[i][3] - gp1)
        Ey = abs(test1[i][4] - gp2)
        Ez = abs(test1[i][5] - gp3)

        ###calculate the angle between two vectors

        modexp = sqrt(test1[i][3] * test1[i][3] + test1[i][4] * test1[i][4] + test1[i][5] * test1[i][5])
        modpre = sqrt(gp1 * gp1 + gp2 * gp2 + gp3 * gp3)
        costheta = (test1[i][3] * gp1 + test1[i][4] * gp2 + test1[i][5] * gp3) / (modexp * modpre)

        errordata.append([Ex, Ey, Ez, costheta, test1[i][0], test1[i][1], test1[i][2]])
        print("IN Calculation")

    print("Data all saved")

    sume1 = 0
    sume2 = 0
    sume3 = 0
    sumb1 = 0
    sumb2 = 0
    sumb3 = 0
    sumbtheta = 0

    x = len(errordata)

    for i in range(len(errordata)):
        sume1 = sume1 + errordata[i][0]
        sume2 = sume2 + errordata[i][1]
        sume3 = sume3 + errordata[i][2]
        sumb1 = sumb1 + (test1[i][3]) ** 2
        sumb2 = sumb2 + (test1[i][4]) ** 2
        sumb3 = sumb3 + (test1[i][5]) ** 2

        sumbtheta = sumbtheta + errordata[i][3]

    sumd1 = sqrt(sumb1 / x)
    sumd2 = sqrt(sumb2 / x)
    sumd3 = sqrt(sumb3 / x)

    errordata = zip(*errordata)
    errordata = np.array(errordata)
    print errordata

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = errordata[4]
    y = errordata[5]
    z = errordata[6]
    c = errordata[3]

    ax.scatter(x, y, z, c=c, color='w', cmap=plt.gray())
    plt.show()

    return (sume1 / x) / (sumd1), (sume2 / x) / (sumd2), (sume3 / x) / (sumd3), sumbtheta / x


#################################### Data Plots ##################################	
# Plot the training date points
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


###################data selection#######################################
def data_select():
    ######### Select the data with a small cube ##########
    rawdata = []
    traindata = []

    # reading data from the files
    csv_in1 = csv.reader(open('/media/qc230/data-d2/Qian/Calibration/Data/Exp4_20151102/Config1/Config1.csv', 'rb'),
                         delimiter='\t')

    for row1 in csv_in1:
        rawdata.append(row1)

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    x1 = np.array(x1).tolist()

    for i in range(len(x1)):
        # if x1[i][0] >= -0.1*1.e3 and x1[i][0] <= 0.31*1.e3: #x dimension

        # if x1[i][2] >= -0.3*1.e3 and x1[i][2] <= 0.2*1.e3: #z dimension

        # if x1[i][1] >= -0.0015*1.e3 and x1[i][1] <= 0.0015*1.e3:

        # print i

        force = sqrt(x1[i][3] * x1[i][3] + x1[i][4] * x1[i][4] + x1[i][5] * x1[i][5])

        traindata.append(
            [x1[i][0] * 1.e-3, x1[i][1] * 1e-3, x1[i][2] * 1e-3, x1[i][3] * 1., x1[i][4] * 1., x1[i][5] * 1., force])

    f1 = np.array(traindata)
    f1 = f1.astype(float)

    with open('/media/qc230/data-d2/Qian/Calibration/Data/Exp4_20151102/Config1/newtraindata1.csv', "wb") as f:

        writer = csv.writer(f, delimiter='\t')
        writer.writerows(f1)

    print len(traindata)


###################data selection#######################################
def data_selection(no):
    cc = str(no)
    ######### Select the data within a small cube ##########
    rawdata = []
    traindata = []

    # reading data from the files
    csv_in1 = csv.reader(
        open('/media/qc230/data-d2/Qian/Calibration/Data/Exp4_20151102/Config' + cc + '/Config' + cc + '.csv', 'rb'),
        delimiter='\t')

    for row1 in csv_in1:
        rawdata.append(row1)

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    x1 = np.array(x1).tolist()

    n = 0  # counter1
    m = 0  # counter2

    for i in range(len(x1)):

        if i == 0:
            traindata.append(
                [x1[n][0] * 1.e-3, x1[n][1] * 1e-3, x1[n][2] * 1e-3, x1[n][3] * 1., x1[n][4] * 1., x1[n][5] * 1.,
                 x1[n][6] * 1.])

            print n

        if abs(x1[i][1] - x1[n][1]) >= 40 and abs(x1[i][0] - x1[n][0]) <= 41:

            n = i

            traindata.append(
                [x1[n][0] * 1.e-3, x1[n][1] * 1e-3, x1[n][2] * 1e-3, x1[n][3] * 1., x1[n][4] * 1., x1[n][5] * 1.,
                 x1[n][6] * 1.])

            print n

        elif abs(x1[i][1] - x1[n][1]) >= 42:

            n = i

            m = i - 1

            traindata.append(
                [x1[m][0] * 1.e-3, x1[m][1] * 1e-3, x1[m][2] * 1e-3, x1[m][3] * 1., x1[m][4] * 1., x1[m][5] * 1.,
                 x1[m][6] * 1.])
            traindata.append(
                [x1[n][0] * 1.e-3, x1[n][1] * 1e-3, x1[n][2] * 1e-3, x1[n][3] * 1., x1[n][4] * 1., x1[n][5] * 1.,
                 x1[n][6] * 1.])

            print m, n

    f1 = np.array(traindata)
    f1 = f1.astype(float)

    print len(f1)

    with open(
            '/media/qc230/data-d2/Qian/Calibration/Data/Exp4_20151102/Config' + cc + '/Config' + cc + '_selected3.csv',
            "wb") as f:

        writer = csv.writer(f, delimiter='\t')
        writer.writerows(f1)

    print len(traindata)


def data_selection2(no):
    cc = str(no)
    ######### Select the data within a small cube ##########
    rawdata = []
    traindata = []

    # reading data from the files
    csv_in1 = csv.reader(
        open('/media/qian/data-d2/Qian/Calibration/Data/Exp6_20160104/Config' + cc + '/Config' + cc + '_selected.csv',
             'rb'), delimiter='\t')

    for row1 in csv_in1:
        rawdata.append(row1)

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    x1 = np.array(x1).tolist()

    print len(x1)

    for n in range(len(x1) - 1):

        if x1[n + 1][3] != x1[n][3]:
            traindata.append([x1[n][0], x1[n][1], x1[n][2], x1[n][3] * 1., x1[n][4] * 1., x1[n][5] * 1., x1[n][6] * 1.])

            print n

    f1 = np.array(traindata)
    f1 = f1.astype(float)

    print len(f1)

    with open('/media/qian/data-d2/Qian/Calibration/Data/Exp6_20160104/Config' + cc + '/Config' + cc + '_selected2.csv',
              "wb") as f:

        writer = csv.writer(f, delimiter='\t')
        writer.writerows(f1)
    ###############################################################################################################


##### Evaluation Group ######
def checkexp():
    # Save the raw data from the experimental measurement
    rawdata = []
    # rawdata1=[]
    # reading data from the files
    csv_in1 = csv.reader(open('/group/data-d2/Qian/Calibration/Data/Exp2_20150730/Config4/training.csv', 'rb'),
                         delimiter='\t')
    # csv_in2 = csv.reader(open('/group/data-d2/Qian/Calibration/Data/Exp2_20150730/Config4/valid.csv', 'rb'), delimiter = '\t')

    # extract data from the csv file
    for row1 in csv_in1:
        rawdata.append(row1)

    # for row2 in csv_in2:
    # rawdata1.append(row2)

    # convert the data type to float
    x1 = np.array(rawdata)
    x1 = x1.astype(float)
    # x1 = np.array(x1).tolist()

    # x1 = rawdata

    print len(x1)

    mod_trans = 0.70710678118  # sqrt(2)/2
    data = []

    for i in range(len(x1)):
        x = x1[i][0] - 100

        y = x1[i][1]

        z = x1[i][2] + 100

        xx = (mod_trans * x + mod_trans * z)

        yy = y

        zz = ((-1) * mod_trans * x + mod_trans * z)

        data.append([xx, yy, zz])

    with open('/group/data-d2/Qian/Calibration/Codes/Learning/checkexp1.csv', "wb") as f:

        writer = csv.writer(f, delimiter='\t')

        writer.writerows(data)


def makeforcedata(no, z_center):
    cc = str(no)

    ####################Gaussian Process (GP)######################
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

            ##################this block rotates the x,z coordinates about y if necessary#############
            # xx = (mod_trans * x + (-1)*mod_trans * z) + 0.100

            # yy = y

            # zz = (mod_trans * x + mod_trans * z) - 0.100
            ##############################################################################

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


##comsol, = plt.plot(el1[0], el1[6], 'r-')
# monopole, = plt.plot(el2[0], el2[6], 'b-')
##plt.legend([comsol, monopole], ['comsol', 'monopole'])
# plt.title('Magnetic force data comparison')
# plt.grid()

####Give a configuration number####	
def validation(no):
    cc = str(no)

    testdata1 = []
    testdata2 = []
    errordata = []

    # reading data from the files
    csv_in1 = csv.reader(open('/group/data-d2/Qian/Calibration/Data/Exp2_20150722/Config' + cc + '/valid.csv', 'rb'),
                         delimiter='\t')

    # extract data from the csv file
    for row1 in csv_in1:
        testdata1.append(row1)

    x1 = np.array(testdata1)
    train1 = x1.astype(float)
    test1 = np.array(train1).tolist()

    gpx = joblib.load('./GP_model/config' + cc + 'x.pkl')
    gpy = joblib.load('./GP_model/config' + cc + 'y.pkl')
    gpz = joblib.load('./GP_model/config' + cc + 'z.pkl')

    ###########Error Analysis################
    for i in range(len(test1)):
        ###making predictions
        gp1, sigma1 = gpx.predict([test1[i][0], test1[i][1], test1[i][2]], eval_MSE=True)
        gp2, sigma2 = gpy.predict([test1[i][0], test1[i][1], test1[i][2]], eval_MSE=True)
        gp3, sigma3 = gpz.predict([test1[i][0], test1[i][1], test1[i][2]], eval_MSE=True)

        ###calculating the error

        Ex = abs(test1[i][3] - gp1)
        Ey = abs(test1[i][4] - gp2)
        Ez = abs(test1[i][5] - gp3)

        ###calculate the angle between two vectors

        modexp = sqrt(test1[i][3] * test1[i][3] + test1[i][4] * test1[i][4] + test1[i][5] * test1[i][5])
        modpre = sqrt(gp1 * gp1 + gp2 * gp2 + gp3 * gp3)
        costheta = (test1[i][3] * gp1 + test1[i][4] * gp2 + test1[i][5] * gp3) / (modexp * modpre)

        errordata.append([Ex, Ey, Ez, costheta, test1[i][0], test1[i][1], test1[i][2]])
        print("IN Calculation")

    print("Data all saved")

    sume1 = 0
    sume2 = 0
    sume3 = 0
    sumb1 = 0
    sumb2 = 0
    sumb3 = 0
    sumbtheta = 0

    x = len(errordata)

    for i in range(len(errordata)):
        sume1 = sume1 + errordata[i][0]
        sume2 = sume2 + errordata[i][1]
        sume3 = sume3 + errordata[i][2]
        sumb1 = sumb1 + (test1[i][3]) ** 2
        sumb2 = sumb2 + (test1[i][4]) ** 2
        sumb3 = sumb3 + (test1[i][5]) ** 2

        sumbtheta = sumbtheta + errordata[i][3]

    sumd1 = sqrt(sumb1 / x)
    sumd2 = sqrt(sumb2 / x)
    sumd3 = sqrt(sumb3 / x)

    errordata = zip(*errordata)
    errordata = np.array(errordata)
    print errordata

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = errordata[4]
    y = errordata[5]
    z = errordata[6]
    c = errordata[3]

    ax.scatter(x, y, z, c=c, color='w', cmap=plt.gray())
    plt.show()

    return (sume1 / x) / (sumd1), (sume2 / x) / (sumd2), (sume3 / x) / (sumd3), sumbtheta / x


#####Base Functions#######	
def Fvin(I):
    if I != 0:

        x = math.sinh(I) / math.cosh(I);  # ; coth(I)-(1/I);

    else:

        x = 0;

    return x


def xrange(start, stop, step):
    while start < stop:
        yield start
        start += step


def tup2float(tup):
    return float('.'.join(str(x) for x in tup))


def f(x):
    """The function to predict."""
    return x * np.sin(x)
