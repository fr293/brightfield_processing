import numpy as np
from pyfiglet import Figlet
import time
from Tkinter import *
# import Tkinter
# import Tkconstants
import tkFileDialog
import matplotlib.pyplot as plt

f = Figlet(font='isometric1')
print f.renderText('Ferg')
time.sleep(1)
g = Figlet(font='isometric3')
print g.renderText('Labs')
time.sleep(1)

print('Welcome to the brightfield bead tracking and processing tool.')
time.sleep(0.5)
print('Please select the folder containing your data ready for analysis:')

root = Tk()
root.attributes("-topmost", True)
root.directory = tkFileDialog.askdirectory()
input_folder = root.directory + '/'

print('please select the file that contains details of the data to be processed:')

root.filename = tkFileDialog.askopenfilename(initialdir='C:/', title='Select file',
                                             filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

fig, axes = plt.subplots(2, 2)

# read in two data sets

# on seperate plots, put traces for N, S, E, W directions, all four

