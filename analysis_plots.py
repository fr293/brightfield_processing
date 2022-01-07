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


def selectFile():
    # takes no parameters, prompts the user to browse for a CSV file in the run directory
    # reads the file
    # returns a list of subfolder filepaths
    return


def obtainRunData(run_filepaths):
    # takes a list of filepaths to run subfolders
    # in each subfolder, find a parameter file and an analysis file
    # returns lists of parameter objects and analysis objects
    return

# point shapes; North: square 's', South: diamond 'D', East: triangle '^', West: inverted triangle 'v'
# plot colours; 1st: red 'r', 2nd: blue 'b', 3rd: green 'g', 4th: yellow 'y'


def plotTerminalDisplacement(run_filepaths):
    # create a figure
    # for each run filepath, extract parameter and analysis data
    # for each duration [10,30,90]:
    #   make a subset of the relevant parameter data, selecting from the duration column
    #   for each configuration:
    #       extract the force and terminal displacement values from the analysis file
    #       plot the points with markers according to the configuration and plot colours
    # make any configuration changes to the figure
    # return the figure
    return


def plotResidualDisplacement():
    return


def plotEtaBeta():
    return


def plotBetaCbeta():
    return

def saveFigure(figureHandle):
    # takes a figure object and saves it
    return