import numpy as np
from pyfiglet import Figlet
import time
from scipy import interpolate
from Tkinter import *
import tkFileDialog
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5, rc={'text.usetex': True, 'font.family': 'serif', 'font.serif': 'cm'})
sns.set_context("talk")
sns.set_style('darkgrid')

experimental_period = 0.5
resample_value = 500

f = Figlet(font='isometric1')
print f.renderText('Ferg')
time.sleep(1)
g = Figlet(font='isometric3')
print g.renderText('Labs')
time.sleep(1)

plot_info = ['sampletrace_1', ["G:\\Shared drives\\Team Backup\\Vesicles Fergus' data\\compaction_experiments\\D_series\\D1\\D1_W_1A_90.csv", 0, 4],
             ["G:\\Shared drives\\Team Backup\\Vesicles Fergus' data\\compaction_experiments\\D_series\\D1\\D1_E_1A_90.csv", 0, 4]]

# print('Welcome to the brightfield bead tracking and processing tool.')
# time.sleep(0.5)
# print('Please select the folder containing your data ready for analysis:')
#
# root = Tk()
# root.attributes("-topmost", True)
# root.directory = tkFileDialog.askdirectory()
# input_folder = root.directory + '/'
#
# print('please select the file that contains details of the data to be processed:')
#
# root.filename = tkFileDialog.askopenfilename(initialdir='C:/', title='Select file',
#                                              filetypes=(('csv files', '*.csv'), ('all files', '*.*')))
#
# fig, axes = plt.subplots(2, 2)

# read in the datasets to be plotted

# generate a window with checkboxes for all of the data series

# choose which series should be plotted

# choose the data from the series to plot

# press the plot button to generate a graph

# load in data
# extract series name
series_name = plot_info[0]
plot_info = plot_info[1:]

# append data to each sublist
for info in plot_info:
    filename, x_data, y_data = info
    extract = np.genfromtxt(filename, dtype=np.float, skip_header=1, delimiter=',',
                       encoding=None, usecols=(x_data, y_data))
    info.append(extract)

# assuming x points are reasonably comparable, concatenate y information

for info in plot_info:



# create plot

fig = plt.figure()

#take in
xnew = np.linspace(0, (experimental_period * x.size) - 1, resample_value)

def resample(x, y, xnew):
    f_interp = interpolate.interp1d(x,y)
    ynew = f_interp(xnew)
    return ynew