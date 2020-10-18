from tkinter import Tk
from gui import NicerGui


if __name__ == '__main__':

    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    nicer_gui = NicerGui(root, (screen_width, screen_height))
    root.lift()
    root.mainloop()



