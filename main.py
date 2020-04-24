from imports import *
from utils import *
from nicer import NICER
import config
from tkinter import Tk
from gui import NicerGui


if __name__ == '__main__':

    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    nicer_gui = NicerGui(root, (screen_width, screen_height))
    root.mainloop()




    
    

# TODOs:

# high priority:
# -- verify RAW support
# -- enable automatic ABN
# -- replace all occurences of 1080 with config.final_size

# medium to low priority:
# -- fix imports
# -- enable CAN7 enhancements?
# -- update GitHub Readme
# -- choose suitable images to upload
# -- resize images so they are smaller in size and require less space
# --
