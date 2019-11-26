import analyse as a
import blobfinder as b
import numpy as np
from pyfiglet import Figlet
import time
from Tkinter import *
# import Tkinter
# import Tkconstants
import tkFileDialog
import preprocess as pre
from os.path import exists
from shutil import copy
import csv

f = Figlet(font='isometric1')
print f.renderText('Ferg')
time.sleep(1)
g = Figlet(font='isometric3')
print g.renderText('Labs')
time.sleep(1)

print('Welcome to the brightfield validation tool.')
time.sleep(0.5)
print('Please select the folder containing your input data:')

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

print('please select the file that contains details of the data to be processed:')

root.filename = tkFileDialog.askopenfilename(
    initialdir='G:\\Shared drives\\Team Backup', title='Select Experiment file',
    filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

# check that all the files mentioned in the input data file exist. If not, then raise a warning and possibly halt

file_array = np.genfromtxt(root.filename, dtype=None, skip_header=1, delimiter=',', encoding=None)
file_list = file_array.tolist()

print('please select the folder where the processed data is saved:')

root.directory = tkFileDialog.askdirectory(
    initialdir='G:/Shared drives/Team Backup',
    title='Select Output Directory')
output_folder = root.directory + '/'

for experiment_run in file_list:
    # extract current configurations
    [filename, ca, cc, fon, fdur, num_frames, frame_period, temp] = experiment_run
    filename = str(filename.replace('"', ''))
    if exists(output_folder + filename + '.csv'):
        print('experiment ' + filename + ' processed, proceeding to analysis')
        experiment_data = np.genfromtxt(output_folder + filename + '.csv', dtype=None, skip_header=1, delimiter=',',
                                   encoding=None)
        position = experiment_data[:, 1:3]
        position = position - position[0, :]
        position_magnitude = np.linalg.norm(position[1:, :], axis=1)
        position_magnitude = position_magnitude[:, np.newaxis]
        position_norm = position[1:, :] / position_magnitude
        position_norm = np.vstack(([0, 0], position_norm))
        force = experiment_data[:, 5:7]
        dotted = np.sum(force*position_norm, 1)



    else:
        print('error')

