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

image_registration = False

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
root.directory = tkFileDialog.askdirectory(initialdir='D:\\sync_folder\\experiments_DNA_Brushes',
                                           title='Select Input Data Directory')
input_filepath = root.directory + '/'

print('please select the file that contains details of the data to be processed:')

root.filename = tkFileDialog.askopenfilename(
    initialdir=input_filepath, title='Select Experiment file',
    filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

# check that all the files mentioned in the input data file exist. If not, then raise a warning and possibly halt

file_array = np.genfromtxt(root.filename, dtype=None, skip_header=1, delimiter=',', encoding=None)
file_list = file_array.tolist()

print('please select the folder that you want the processed data to be saved in:')

root.directory = tkFileDialog.askdirectory(
    initialdir='G:\\Shared drives\\vesicles_team\\fergus_data', title='Select Output Directory')
output_filepath = root.directory + '/'

# copy the experimental file over
try:
    copy(root.filename, output_filepath)
except IOError:
    print ('the experimental file already exists in the output folder')

# file mode 'wb' will overwrite file contents if one exists
with open(output_filepath + 'experiment_analysis.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Run Name', 'Peak Deformation', 'Peak Force', 'Residual Deformation', 'Model Plasticity', 'Eta',
                     'C Beta', 'Beta'])

print('success: starting analysis')


# run through each experiment, extracting data and analysing where necessary 
for experiment_run in file_list:
    # extract current configurations
    [filename, ca, cc, fon, fdur, num_frames, frame_period, temp] = experiment_run
    filename = str(filename.replace('"', ''))

    if image_registration:
        print('using image registration')
        if exists(input_filepath + filename + '_r.tiff'):
            print('experiment ' + filename + ' registered, proceeding to analysis')
        else:
            print('preprocessing experiment: ' + filename)
            try:
                pre.register(filename + '_r.tiff', input_filepath, filename + '.tiff', input_filepath)
            except IOError:
                print('error: experimental image data not found')
                continue
    else:
        print('proceeding without image registration')
        if exists(input_filepath + filename + '.tiff'):
            print('experiment ' + filename + ' found, proceeding to data extraction')
        else:
            print('error: experimental image data not found')

    if not (exists(output_filepath + filename + '.csv') and exists(output_filepath + filename + '_rheos.csv')):
        try:
            [time_stack, position_stack, force_stack, force_mask, eigenforce, eigendisplacement,
             alignment_ratio] = b.findblobstack(filename, input_filepath, output_filepath, ca, cc)

            # unit normalisation
            position_stack = position_stack * 1E-6
            eigendisplacement = eigendisplacement * 1E-6
            force_stack = force_stack * 1E-9
            eigenforce = eigenforce * 1E-9

            exp_data = np.hstack([time_stack, position_stack, eigendisplacement, alignment_ratio, force_stack,
                                  eigenforce, force_mask])
            exp_data_rheos = np.hstack([time_stack, eigendisplacement, eigenforce])

            np.savetxt(output_filepath + filename + '_full.csv', exp_data, delimiter=',',
                       header='time/s,position x/m,position y/m,position z/m,distance along force vector/m,'
                              'alignment ratio,x mean force estimate/N,y mean force estimate/N,'
                              'z mean force estimate/N, plane force/N,force on/off')

            np.savetxt(output_filepath + filename + '_rheos.csv', exp_data_rheos, delimiter=',',
                       header='time/s, displacement/m, force/N')

        except IOError:
            print('error: experimental time data not found')
    else:
        print('experimental data exists, proceeding to next experiment')



    print('postprocessing experiment: ' + filename)
        try:
        analysed_data = a.full_analysis(filename, output_filepath)
        analysed_data = [filename] + list(analysed_data)

# the file mode ab appends to the end of the file
        with open(output_filepath + 'experiment_analysis.csv', 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(analysed_data)

    except IndexError:
        print('error: not able to postprocess data for ' + filename)
        error_message = [filename] + ['warning', 'no', 'data', 'available', '!']

        with open(output_filepath + 'experiment_analysis.csv', 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(error_message)}

# train distributions based on extracted parameters 


# if the training is successful, go over each experiment again and compute prediction bounds 
for experiment_run in file_list:
    # extract current configurations
    [filename, ca, cc, fon, fdur, num_frames, frame_period, temp] = experiment_run
    filename = str(filename.replace('"', ''))


