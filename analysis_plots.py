# we want to take in a all the runs from one batch at a time and plot several check graphs
# batches are contained within a parent folder, and the child folders all contain one run
# each child folder contains one list of experimental parameters and one list of experimental outcomes

# strategy:
# user select the file containing the batch run data. This file is in the parent folder at the top level
# for each entry in the batch run data file, obtain the experimental parameter and analysis files
# for each type of analysis plot, search through the parameter file to find relevant experiments
# produce each plot, saving in the parent folder at the top level

# parameter files hold data in this format:
# Run Name, Amplitude, Configuration, Force on, Duration, number of frames, frame period, Temperature
# Run Name corresponds to the entry in the analysis file
# Amplitude: (1 - 0.5A, 2 - 1.0A, 3 - 1.5A, 4 - 2.0A, 5 - 2.5A), (6 - 0.1A, 7 - 0.2A, 8 - 0.3A, 9 - 0.4A)
# Configuration 1 - W, 2 - S, 3 - N, 4 - E
# force on is set at 10s
# Duration varies as [10, 30, 90]s
# Number of Frames and Frame period are deprecated parameters
# Temperature is unimplemented as of this release

# analysis files hold data in this format:
# Run Name,	Peak Deformation, Peak Force, Residual Deformation, Model Plasticity, Eta, C Beta, Beta, Optimiser
# Run Name corresponds to the entry in the parameter file
# Peak Deformation is the absolute deviation in position from the starting point, measured at the last image in the
# creep region
# Peak force is the force value coressponding to the Peak Deformation position coordinates
# Residual Deformation is the absolute deviation in position from the starting point, averaged over at the last few
# images in the recovery region
# Model Plasticity is the size of Residual deformation as predicted by the fitted model
# Eta, C Beta and Beta are the model parameters, as fitted by a springpot or fractional Maxwell
# Optimiser reports the RHEOS optimisation algorithm that produced the lowest error in fitting

import numpy as np
import pathlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Tkinter import *
import tkFileDialog

def selectFile():
    # takes no parameters, prompts the user to browse for a CSV file in the run directory
    # reads the file
    # returns a list of subfolder filepaths and associated sample names
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.filename = tkFileDialog.askopenfilename(
        initialdir='/Volumes/GoogleDrive/Shared drives/vesicles_team/fergus_data', title='Select Series file',
        filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

    filepath = pathlib.Path(root.filename)
    save_folder = filepath.parent

    file_array = np.genfromtxt(root.filename, dtype=None, delimiter=',', encoding=None)
    file_paths = file_array[:,0]
    run_names = file_array[:,1]
    run_filepaths = file_paths.tolist()
    run_name_list = run_names.tolist()

    return run_filepaths, run_name_list, save_folder


def obtainRunData(run_filepaths):
    # takes a list of filepaths to run subfolders
    # in each subfolder, find a parameter file and an analysis file
    # returns lists of parameter arrays and analysis arrays
    parameter_arrays = []
    analysis_arrays = []

    for filepath in run_filepaths:
        # this is a slightly shonky way of dealing with the variability in file naming
        parameter_list = list(pathlib.Path(filepath).glob('*exp_file.csv'))
        analysis_list = list(pathlib.Path(filepath).glob('*experiment_analysis.csv'))
        if parameter_list and analysis_list:
            parameter_arrays.append(np.genfromtxt(parameter_list[0], dtype=None, skip_header=1, delimiter=',',
                                    encoding=None))
            analysis_arrays.append(np.genfromtxt(analysis_list[0], dtype=None, skip_header=1, delimiter=',',
                                    encoding=None))
        else:
            parameter_arrays.append([])
            analysis_arrays.append([])

    return parameter_arrays, analysis_arrays

# point shapes; North: square 's', South: diamond 'D', East: triangle '^', West: inverted triangle 'v'
# plot colours; 1st: red 'r', 2nd: blue 'b', 3rd: green 'g', 4th: yellow 'y'


def displacementPlotGenerator(parameter_arrays, analysis_arrays, run_name_list, save_folder):
    # plot types: force and terminal displacement, force and residual displacement
    # for each duration [10,30,90]:
    #   make a subset of the relevant parameter data, selecting from the duration column
    #   for each configuration:
    #       extract the force and terminal/residual displacement values from the analysis file
    #       plot the points with markers according to the configuration and plot colours
    # make formatting changes to the figure
    # return the figure

    colours = ['r', 'b', 'g', 'y', 'c', 'm']
    markers = ['v', 'D', 's', '^']
    durations = ['10s', '30s', '90s']

    terminal_10_fig, terminal_10_ax = plt.subplots()
    terminal_30_fig, terminal_30_ax = plt.subplots()
    terminal_90_fig, terminal_90_ax = plt.subplots()
    residual_10_fig, residual_10_ax = plt.subplots()
    residual_30_fig, residual_30_ax = plt.subplots()
    residual_90_fig, residual_90_ax = plt.subplots()

    summary_fig, summary_axs = plt.subplots(2, 3, sharex='all', sharey='row')

    terminal_axes_list = [terminal_10_ax, terminal_30_ax, terminal_90_ax]
    residual_axes_list = [residual_10_ax, residual_30_ax, residual_90_ax]

    for run in range(len(parameter_arrays)):
        colour = colours[run]
        parameter_entry = parameter_arrays[run]
        analysis_entry = analysis_arrays[run]

        force_10_list = []
        force_30_list = []
        force_90_list = []

        terminal_10_list = []
        terminal_30_list = []
        terminal_90_list = []

        residual_10_list = []
        residual_30_list = []
        residual_90_list = []

        for experiment in range(len(analysis_entry)):
            parameters = parameter_entry[experiment]
            analysis = analysis_entry[experiment]

            if parameters[4] == 10:
                force_10_list.append(analysis[2])
                terminal_10_list.append(analysis[1])
                residual_10_list.append(analysis[3])

            if parameters[4] == 30:
                force_30_list.append(analysis[2])
                terminal_30_list.append(analysis[1])
                residual_30_list.append(analysis[3])

            if parameters[4] == 90:
                force_90_list.append(analysis[2])
                terminal_90_list.append(analysis[1])
                residual_90_list.append(analysis[3])
    # update plots
        force_10_array = (np.array(force_10_list)) * (10 ** 9)
        force_30_array = (np.array(force_30_list)) * (10 ** 9)
        force_90_array = (np.array(force_90_list)) * (10 ** 9)

        terminal_10_array = (np.array(terminal_10_list)) * (10 ** 6)
        terminal_30_array = (np.array(terminal_30_list)) * (10 ** 6)
        terminal_90_array = (np.array(terminal_90_list)) * (10 ** 6)

        residual_10_array = (np.array(residual_10_list)) * (10 ** 6)
        residual_30_array = (np.array(residual_30_list)) * (10 ** 6)
        residual_90_array = (np.array(residual_90_list)) * (10 ** 6)

        terminal_10_ax.scatter(force_10_array, terminal_10_array, c=colour)
        terminal_30_ax.scatter(force_30_array, terminal_30_array, c=colour)
        terminal_90_ax.scatter(force_90_array, terminal_90_array, c=colour)
        summary_axs[0, 0].scatter(force_10_array, terminal_10_array, c=colour)
        summary_axs[0, 1].scatter(force_30_array, terminal_30_array, c=colour)
        summary_axs[0, 2].scatter(force_90_array, terminal_90_array, c=colour)

        residual_10_ax.scatter(force_10_array, residual_10_array, c=colour)
        residual_30_ax.scatter(force_30_array, residual_30_array, c=colour)
        residual_90_ax.scatter(force_90_array, residual_90_array, c=colour)
        summary_axs[1, 0].scatter(force_10_array, residual_10_array, c=colour)
        summary_axs[1, 1].scatter(force_30_array, residual_30_array, c=colour)
        summary_axs[1, 2].scatter(force_90_array, residual_90_array, c=colour)

    for duration in range(len(terminal_axes_list)):
        axes = terminal_axes_list[duration]
        title_string = 'Maximum Displacement, ' + durations[duration] + ' Creep'
        axes.set_title(title_string)
        axes.set_xlabel('Creep Force/ nN')
        axes.set_ylabel('Maximum Displacement/ um')
        axes.legend(run_name_list, loc='upper left', fontsize='medium', framealpha=0.5)

    for duration in range(len(residual_axes_list)):
        axes = residual_axes_list[duration]
        title_string = 'Residual Displacement, ' + durations[duration] + ' Creep'
        axes.set_title(title_string)
        axes.set_xlabel('Creep Force/ nN')
        axes.set_ylabel('Residual Displacement/ um')
        axes.legend(run_name_list, loc='upper left', fontsize='medium', framealpha=0.5)

    summary_fig.text(0.52, 0.03, 'Terminal Force/ nN', ha='center')
    summary_fig.text(0.02, 0.6, 'Displacement/ um', ha='center', rotation='vertical')
    summary_axs[0, 0].legend(run_name_list, loc='upper left', fontsize='small', framealpha=0.5)
    summary_axs[0, 0].set_ylabel('Maximum')
    summary_axs[1, 0].set_ylabel('Plastic')
    summary_axs[0, 0].set_title('10s Creep')
    summary_axs[0, 1].set_title('30s Creep')
    summary_axs[0, 2].set_title('90s Creep')

    terminal_10_fig.savefig(str(save_folder.joinpath('t_figure_10.pdf')))
    terminal_30_fig.savefig(str(save_folder.joinpath('t_figure_30.pdf')))
    terminal_90_fig.savefig(str(save_folder.joinpath('t_figure_90.pdf')))
    residual_10_fig.savefig(str(save_folder.joinpath('r_figure_10.pdf')))
    residual_30_fig.savefig(str(save_folder.joinpath('r_figure_30.pdf')))
    residual_90_fig.savefig(str(save_folder.joinpath('r_figure_90.pdf')))

    summary_fig.suptitle('Maximum and Plastic Creep Displacements')
    summary_fig.savefig(str(save_folder.joinpath('summary.pdf')))

    plt.close('all')

    return


def plotEtaBeta():
    # define the plot types:
    # eb: eta and beta, bcb: beta and cbeta
    # create a figure
    # for each run filepath, extract parameter and analysis data
    # for each duration [10,30,90]:
    #   make a subset of the relevant parameter data, selecting from the duration column
    #   for each configuration:
    #       extract the material parameters from the analysis file
    #       plot the points with markers according to the configuration and plot colours
    # make formatting changes to the figure
    # return the figure
    return


def wrapper():
    run_filepaths, run_name_list, save_folder = selectFile()

    parameter_arrays, analysis_arrays = obtainRunData(run_filepaths)

    displacementPlotGenerator(parameter_arrays, analysis_arrays, run_name_list, save_folder)

    return