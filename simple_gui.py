from Tkinter import *
import tkFileDialog

root = Tk()

col_sep = "\t"
col_h_b = []  # field column for background
col_m_b = []  # magnetization column for background


def choose_b_file(self):
    root.fileName_b = tkFileDialog.askopenfilename(filetypes=((".csv files", "*.csv"), ("All files", "*.*")))


def plot():
    if offset.get() == 1:
        print('True')
        # some mathematical operations and graph plotting
    else:
        print('False')
        # other mathematical operations and graph plotting


def close_window():
    exit()


filename = StringVar()
w = Label(root, text=filename)
w.pack()

offset = IntVar()
checkbox = Checkbutton(root, text="offset subtraction", variable=offset)
checkbox.pack()

b_data = Button(root, text="Background", width=20, command=choose_b_file())
m_minus_b = Button(root, text="Plot", width=5, command=plot)
quit = Button(root, text="Quit", width=5, command=close_window)

b_data.pack()
m_minus_b.pack()
quit.pack()

root.mainloop()
