

from data_processing import *
from scrapper import *
from plot_functions import *
from tkinter import *
from evaluation import*


if __name__ == '__main__':
    root = Tk()
    root.title('Application')
    root.geometry("500x450")

    Label(root, text="Step 2) Process input datasets", font='Helvetica 10 bold').grid(row=3, column=1, columnspan=2, sticky=W, pady=4)
    Button(root, text='Process input datasets', command=process_data, width=30).grid(row=4, column=1, sticky=W, pady=4)

    root.mainloop()
