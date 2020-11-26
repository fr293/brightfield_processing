import analyse as a
import blobfinder as b
import numpy as np
from pyfiglet import Figlet
import time
import Tkinter
# import Tkconstants
import tkFileDialog
import tkMessageBox
import preprocess as pre
from os.path import exists
from shutil import copy
import csv


class process_tk(Tkinter.Tk):
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()

        self.entryVariable = Tkinter.StringVar()

        self.entry = Tkinter.Entry(self,textvariable=self.entryVariable)
        self.entry.grid(column=0,row=0,sticky='EW')
        self.entry.bind("<Return>", self.OnPressEnter)
        self.entryVariable.set(u"Enter text here.")

        button_browse = Tkinter.Button(self, text = "Browse", command = lambda:self.entryVariable.set(tkFileDialog.askopenfilename()))


        button = Tkinter.Button(self,text="Go",
                                 command=self.OnButtonClick)
        button_browse.grid(column=1,row=0)
        button.grid(column=2,row=0)
        offset = Tkinter.IntVar()
        checkbox = Tkinter.Checkbutton(self, text="offset subtraction", variable=offset)
        checkbox.grid(column=3, row = 0)

        self.labelVariable = Tkinter.StringVar()
        label = Tkinter.Label(self,textvariable=self.labelVariable,
                              anchor="w",fg="black",bg="white")
        label.grid(column=0,row=1,columnspan=3,sticky='EW')
        self.labelVariable.set(u"Hello !")

        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,False)
        self.update()
        self.geometry(self.geometry())
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

        f = Figlet(font='isometric1')
        print f.renderText('Ferg')
        time.sleep(1)
        g = Figlet(font='isometric3')
        print g.renderText('Labs')
        time.sleep(1)

    def preprocess(self):
        # register the images in each run
        return

    def process(self):
        # find the bead location and forces in each run, saving to a data file
        return

    def postprocess(self):
        # extract key parameters from all of an experiment's data files, saving the result to a new data file
        # produce key plots of an experiment's summary statistics
        return

    def OnButtonClick(self):
        self.labelVariable.set( self.entryVariable.get()+" (You clicked the button)" )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def OnPressEnter(self,event):
        self.labelVariable.set( self.entryVariable.get()+" (You pressed ENTER)" )
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)


if __name__ == "__main__":
    app = process_tk(None)
    app.title('brightfield processing tool')
    app.mainloop()