# this file processes the data as it comes in from the microscope computer
# it assumes that the data is held in a folder called sweep_data, in subfolders numbered 1-5

import os
import numpy as np
import imageio
from pystackreg import StackReg
from shutil import copy2
from tqdm import tqdm
# import csv
from pyfiglet import Figlet
import time
from Tkinter import *
import tkFileDialog


# rename the files that hold the data from dat to csv
# loop through the folders, renaming each file to a csv in turn
def process_tracks(filepath, old_str, new_str):
    i = 0
    for path, subdirs, files in os.walk(filepath):
        for name in files:
            if old_str.lower() in name.lower():
                os.rename(os.path.join(path, name), os.path.join(path, name.lower().replace(old_str, new_str)))
                i = i + 1
    return i


# collate images together, changing the output of the data capture program from single images to stacks and moving the
# associated timing files into a single folder, renaming as appropriate.
def collate(collated_filename, collated_filepath, raw_filepath):
    try:
        os.mkdir(collated_filepath)
        print('directory made, collating ' + collated_filename)
    except:
        print('directory exists, collating ' + collated_filename)
    with imageio.get_writer(collated_filepath + collated_filename + '.tiff') as stack:
        for i in tqdm(range(541)):
            filename = raw_filepath + '/cam1/' + '/cam1_' + str(i + 101) + '.tiff'
            stack.append_data(imageio.imread(filename))
    try:
        time_data = raw_filepath + '/cam1/' + collated_filename + '_time.csv'
        copy2(time_data, collated_filepath)
    except:
        print('could not find time data')


# register images together, taking a stack and saving another registered stack in its place.
def register(registered_filename, registered_filepath, raw_filename, raw_filepath):
    image_stack = imageio.volread(raw_filepath + raw_filename)
    sr = StackReg(StackReg.RIGID_BODY)
    registered_data = sr.register_transform_stack(image_stack, reference='first', verbose=True)
    finished_file = np.uint8(registered_data)
    try:
        os.mkdir(registered_filepath)
        print('directory made, saving registered image:' + registered_filename)
    except:
        print('directory exists, saving registered image:' + registered_filename)
    imageio.volwrite(registered_filepath + '/' + registered_filename, finished_file, bigtiff=True)

# take in an input directory and an output directory, extract uncollated images from the input directory, collate and
# register them and save in the output directory. The folders with the raw data are specified by a csv file in the input
# directory, which is appended to and saved to the output directory. The csvs containing the timings for each run are
# also copied over to the output directory.
def process_images():
    f = Figlet(font='isometric1')
    print f.renderText('Ferg')
    time.sleep(1)
    g = Figlet(font='isometric3')
    print g.renderText('Labs')
    time.sleep(1)

    print('Welcome to the brightfield bead image data preprocessing tool.')
    time.sleep(0.5)
    print('Please select the folder containing your input data:')

    root = Tk()
    root.attributes("-topmost", True)
    root.directory = tkFileDialog.askdirectory()
    input_folder = root.directory + '/'

    print('please select the file that contains details of the data to be processed:')

    root.filename = tkFileDialog.askopenfilename(initialdir='C:/', title='Select file',
                                                 filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

    # check that all the files mentioned in the input data file exist. If not, then raise a warning and possibly halt

    file_array = np.genfromtxt(root.filename, dtype=None, skip_header=1, delimiter=',', encoding=None)
    file_list = file_array.tolist()

    print('please select the file that you want the processed data to be saved in:')

    root.directory = tkFileDialog.askdirectory()
    output_folder = root.directory + '/'

    for experiment_run in file_list:
         [filename, ca, cc, tem] = experiment_run
         collate(filename, input_folder + '/' + filename + '/collated/', input_folder + '/' + filename)

    for experiment_run in file_list:
        [filename, ca, cc, tem] = experiment_run
        register(filename, output_folder, filename + '.tiff', input_folder + '/' + filename + '/collated/')
