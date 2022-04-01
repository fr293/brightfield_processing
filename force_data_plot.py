# plots combined force sampling data for a particular magnitude

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# import the csv files

pathPrefix = '/Users/fergusriche/Library/CloudStorage/OneDrive-UniversityofCambridge/force_sampling_plots/' \
                'noam_1A_171019_mod_I_config_'

config = [1, 2, 3, 4]
colours = ['Purples', 'Blues', 'Reds', 'Greens']
sampleData = []

for i in config:
    fullPath = pathPrefix + str(i) + '.csv'
    extractData = np.array(
        np.genfromtxt(fullPath, dtype=float, usecols=(0, 1, 2, 3, 4, 5, 12), delimiter=',', skip_header=1))

    sampleData.append(extractData)

# create 2by2 plot
fig = plt.figure()

a = 0.5
p = 0.25

# 3 plots are orthographic views

axXY = fig.add_subplot(2, 2, 3)
axXZ = fig.add_subplot(2, 2, 1, sharex=axXY)
axZY = fig.add_subplot(2, 2, 4, sharey=axXY)
ax3D = fig.add_subplot(2, 2, 2, projection='3d')

for i in [0,3]:
    data = sampleData[i]
    data = data[np.random.randint(data.shape[0], size=np.int(p*data.shape[0])), :]

    # XY plot
    axXY.scatter(data[:, 0], data[:, 1], s=data[:, 6]**2, c=data[:, 6], cmap=colours[i], alpha=a)
    axXY.set_xlabel('x um')
    axXY.set_ylabel('y um')
    # XZ plot
    axXZ.scatter(data[:, 0], data[:, 2], s=data[:, 6]**2, c=data[:, 6], cmap=colours[i], alpha=a)
    axXZ.set_ylabel('z um')
    # ZY plot
    axZY.scatter(data[:, 2], data[:, 1], s=data[:, 6]**2, c=data[:, 6], cmap=colours[i], alpha=a)
    axZY.set_xlabel('z um')
    # isotropic view
    ax3D.scatter(data[:, 0], data[:, 1], data[:, 2], s=data[:, 6]**2, c=data[:, 6], cmap=colours[i], alpha=a)

fig.suptitle('Force Sampling Trajectories: 1.0A East and West')

fig.savefig('/Users/fergusriche/Desktop/forcesample.png')
fig.savefig('/Users/fergusriche/Desktop/forcesample.eps')
plt.close(fig)
