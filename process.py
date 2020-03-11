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

print('Welcome to the brightfield bead tracking and processing tool.')
time.sleep(0.5)
print('Please select the folder containing your input data:')

root = Tk()
root.withdraw()
root.attributes("-topmost", True)
root.directory = tkFileDialog.askdirectory(initialdir='D:\\sync_folder', title='Select Input Data Directory')
input_folder = root.directory + '/'

print('please select the file that contains details of the data to be processed:')

root.filename = tkFileDialog.askopenfilename(
    initialdir=input_folder, title='Select Experiment file',
    filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

# check that all the files mentioned in the input data file exist. If not, then raise a warning and possibly halt

file_array = np.genfromtxt(root.filename, dtype=None, skip_header=1, delimiter=',', encoding=None)
file_list = file_array.tolist()

print('please select the file that you want the processed data to be saved in:')

root.directory = tkFileDialog.askdirectory(
    initialdir='G:/Shared drives/Team Backup',
    title='Select Output Directory')
output_folder = root.directory + '/'

# copy the experimental file over
copy(root.filename, output_folder)

with open(output_folder + 'experiment_analysis.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['run name', 'viscous displacement', 'creep viscosity',
                     'creep time', 'residual deformation', 'peak deformation'])
# np.savetxt(output_folder + 'experiment_analysis.csv', ['run name', 'viscous displacement', 'creep viscosity',
#                                                       'creep time', 'residual deformation', 'peak deformation'],
#            fmt='%s', delimiter=',')

print('success: starting analysis')
for experiment_run in file_list:
    # extract current configurations
    [filename, ca, cc, fon, fdur, num_frames, frame_period, temp] = experiment_run
    filename = str(filename.replace('"', ''))
    # if exists(input_folder + filename + '_r.tiff'):
    #     print('experiment ' + filename + ' registered, proceeding to analysis')
    # else:
    #     print('preprocessing experiment: ' + filename)
    #     try:
    #         pre.register(filename + '_r.tiff', input_folder, filename + '.tiff', input_folder)
    #     except IOError:
    #         print('error: experimental image data not found')
    #         continue

    if exists(input_folder + filename + '.tiff'):
        print('experiment ' + filename + ' registered, proceeding to analysis')
    else:
        print('preprocessing experiment: ' + filename)
        try:
            pre.register(filename + '_r.tiff', input_folder, filename + '.tiff', input_folder)
        except IOError:
            print('error: experimental image data not found')
            continue

    #user_input = raw_input('Process ')

    print('analysing experiment: ' + filename)

    # try:
    #     [time_stack, absolute_position_stack, scaled_position_stack, force_mean, force_std, force_mask] =\
    #         b.findblobstack(filename + '_r', input_folder, output_folder, ca, cc, cropsize, blobsize,
    #                         gamma_adjust)
    # except IOError:
    #     print('error: experimental time data not found')

    try:
        [time_stack, absolute_position_stack, scaled_position_stack, force_mean, force_std, force_mask] =\
            b.findblobstack(filename, input_folder, output_folder, ca, cc)
    except IOError:
        print('error: experimental time data not found')

    exp_data = np.hstack([time_stack, absolute_position_stack, scaled_position_stack, force_mean, force_std,
                          force_mask])

    np.savetxt(output_folder + filename + '.csv', exp_data, delimiter=',',
               header='time/s,position x/um,position y/um,position z/um,distance along force vector/um,'
                      'x mean force estimate/nN,y mean force estimate/nN,z mean force estimate/nN,'
                      'x std dev force estimate/nN,y std dev force estimate/nN,z std dev force estimate/nN,'
                      'force on/off')

    print('postprocessing experiment: ' + filename)

    try:
        analysed_data = a.full_analysis(output_folder + filename + '.csv')

        analysed_data = [filename] + list(analysed_data)

        with open(output_folder + 'experiment_analysis.csv', 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(analysed_data)

    except IndexError:
        print('error: not able to extract force data')
        error_message = [filename] + ['warning', 'no', 'data', 'available', '!']

        with open(output_folder + 'experiment_analysis.csv', 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(error_message)
