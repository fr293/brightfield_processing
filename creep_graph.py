from pyfiglet import Figlet
import time
import sys
from Tkinter import *
import tkFileDialog
from os.path import exists
import numpy as np

f = Figlet(font='isometric1')
print f.renderText('Ferg')
time.sleep(1)
g = Figlet(font='isometric3')
print g.renderText('Labs')
time.sleep(1)

print('Welcome to the graphical creep comparison tool ')
time.sleep(0.5)

try:
    series_num = int(input('how many data series are you comparing?'))
except NameError:
    print 'error: input not a number, program exiting'
    sys.exit()

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

input_list = np.array()

for i in range(series_num):
    root.directory = tkFileDialog.askdirectory(initialdir='D:\\sync_folder', title='Select Input Data Directory')
    input_list.append(root.directory + '/')

for item in input_list:
    print item