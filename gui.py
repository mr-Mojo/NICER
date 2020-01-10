from imports import *
from tkinter import Tk, Label, Button, DoubleVar, Scale, HORIZONTAL, PhotoImage, filedialog
import numpy as np
from PIL import Image, ImageTk
import os
import config
from nicer import NICER
import tkinter as tk
from tkinter.ttk import Label, Button

class NicerGui:
    def __init__(self, master, screen_size):
        if screen_size[0] > 1920:
            screen_size = list(screen_size)
            screen_size[0] = 1920 
        self.master = master
        self.height = int(0.85*screen_size[1])
        self.width = int(0.85*screen_size[0])
        master.title("NICER - Neural Image Correction and Enhancement Routine")
        master.geometry(str(self.width) + 'x' + str(self.height))  # let gui be x% of screen, centered
        sliderlength = 200

        self.nicer = NICER(checkpoint_can=config.can_checkpoint_path, checkpoint_nima=config.nima_checkpoint_path)

        # keep references to the images for further processing, bc tk.PhotoImage cannot be converted back
        self.reference_img1 = None
        self.reference_img2 = None
        self.img_namestring = None
        self.img_extension = None

        # labels:
        if True:
            self.saturation_label = Label(master, text="Saturation")
            self.contrast_label = Label(master, text="Contrast")
            self.brightness_label = Label(master, text="Brightness")
            self.shadows_label = Label(master, text="Shadows")
            self.highlights_label = Label(master, text="Highlights")
            self.exposure_label = Label(master, text="Exposure")
            self.locallaplacian_label = Label(master, text="Local Laplacian Filtering")
            self.nonlocaldehazing_label = Label(master, text="Non-Local Dehazing")
            self.gamma_label = Label(master, text="Gamma")
            self.print_label = Label(master, text="Open an image to get started!")
            self.print_label.place(x = int(0.635*self.width), y = int(0.96*self.height))     # TODO noch nicht variabel
            self.slider_labels = [self.saturation_label, self.contrast_label, self.brightness_label, self.shadows_label,
                                  self.highlights_label, self.exposure_label, self.locallaplacian_label, self.nonlocaldehazing_label]

        # sliders:
        if True:
            self.saturation = DoubleVar()
            self.saturation_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.saturation)

            self.contrast = DoubleVar()
            self.contrast_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.contrast)

            self.brightness = DoubleVar()
            self.brightness_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.brightness)

            self.shadows = DoubleVar()
            self.shadows_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.shadows)

            self.highlights = DoubleVar()
            self.highlights_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.highlights)

            self.exposure = DoubleVar()
            self.exposure_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.exposure)

            self.locallaplacian = DoubleVar()
            self.locallaplacian_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.locallaplacian)

            self.nonlocaldehazing = DoubleVar()
            self.nonlocaldehazing_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL, var=self.nonlocaldehazing)

            self.gamma = DoubleVar()
            self.gamma_slider = Scale(master, from_=0.005, to=0.5, length=sliderlength, orient=HORIZONTAL, var=self.gamma, resolution=0.005)
            self.gamma_slider.set(0.1)

            self.slider_variables = [self.saturation, self.contrast, self.brightness, self.shadows,
                                     self.highlights, self.exposure, self.locallaplacian, self.nonlocaldehazing]
            self.sliders = [self.saturation_slider, self.contrast_slider, self.brightness_slider, self.shadows_slider,
                            self.highlights_slider, self.exposure_slider, self.locallaplacian_slider, self.nonlocaldehazing_slider]

        # create images and line, and place them
        if True:
            w = tk.Canvas(master, width=20, height=self.height)
            w.place(x=int(0.365*self.width), y=10)
            w.create_line(10, 20, 10, int(0.9*self.height), fill="#476042", dash=(4,4))

            pil_img_one = Image.new('RGB', (224,224), (255, 255, 255))
            pil_img_two = Image.new('RGB', (224,224), (150, 150, 150))
            tk_img_one = ImageTk.PhotoImage(pil_img_one)
            tk_img_two = ImageTk.PhotoImage(pil_img_two)
            self.tk_img_panel_one = Label(master, image=tk_img_one)
            self.tk_img_panel_two = Label(master, image=tk_img_two)
            self.tk_img_panel_one.image = tk_img_one
            self.tk_img_panel_two.image = tk_img_two
            # image pack happens later when open Image button is clicked

        # place sliders and their labels:
        if True:
            # get 65% of screen height:
            three_quarters = int(0.65*self.height)
            space = (three_quarters-30)/7

            for idx, label in enumerate(self.slider_labels):
                label.place(x=20, y=30 + idx * space)
            for idx, label in enumerate(self.sliders):
                label.place(x=150, y=10 + idx * space)

            gamma_space = 2*space if space<60 else 120
            self.gamma_label.place(x=20, y=50+6*space+gamma_space)
            self.gamma_slider.place(x=150, y=30+6*space+gamma_space)

        # create buttons and place
        if True:
            self.open_button = Button(master, text="Open Image", command=self.open_image)
            self.save_button = Button(master, text="Save Image", command=self.save_image)
            self.reset_button = Button(master, text="Reset", command=self.reset_all)
            self.preview_button = Button(master, text="Preview", command=self.preview)
            self.nicer_button = Button(master, text="NICER!", command=self.get_all_slider_values)

            screen_center = int(self.width/2.0)
            button_y = 50+6*space+gamma_space + 50
            self.open_button.place(x=40, y=button_y)
            self.save_button.place(x=40, y=button_y+40)
            self.nicer_button.place(x=40 + 100, y=button_y+20)
            self.preview_button.place(x=40 + 200, y=button_y)
            self.reset_button.place(x=40 + 200, y=button_y+40)

    def open_image(self):
        filepath = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select an image to open",
                                   filetypes = (("jpg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        if filepath.split('.')[-1] in config.supported_extensions:

            self.img_namestring = filepath.split('.')[0]
            self.img_extension = filepath.split('.')[-1]

            pil_img = Image.open(filepath)
            img_width = pil_img.size[0]
            img_height = pil_img.size[1]

            # resize images so that they can fit next to each other:

            # margins of the display canvas: 0.6*windowwidth, 0.9*windowheight
            if img_width > img_height:
                max_width = int(0.6*self.width)         # images wider than high: place above each other
                max_height = int(0.5*0.9*self.height)   # max height is half the display canvas

            elif img_height > img_width:
                max_width = int(0.5*0.6*self.width)       # images higher than wide: place next to each other
                max_height = int(0.9*self.width)

            max_size = int(0.9*self.height) if img_height>img_width else int(0.6*self.width)

            # just to that there are no errors if image is not resized:
            new_img_width = img_width
            new_img_height = img_height

            # img too large: either exceeds max size, or it is too broad / high to be displayed next to each other
            if max(pil_img.size) > max_size or pil_img.size[1] > max_height or pil_img.size[0] > max_width:
                longer_side = max(pil_img.size)
                factor = max_size / longer_side
                new_img_width = int(img_width*factor)
                new_img_height = int(img_height*factor)

                if img_width > img_height and new_img_height > max_height:      # landscape format
                    while new_img_height > max_height:
                        factor *= 0.99
                        print("reduced factor to %f" % factor)
                        new_img_width = int(img_width * factor)
                        new_img_height = int(img_height * factor)

                elif img_height > img_width and new_img_width > max_width:      # portrait format
                    while new_img_width > max_width:
                        factor *= 0.99
                        print("reduced factor to %f" % factor)
                        new_img_width = int(img_width * factor)
                        new_img_height = int(img_height * factor)
                else:
                   pass
                # TODO: square images?

                # img is now resized in a way for 2 images to fit next to each other
                pil_img = pil_img.resize((new_img_width, new_img_height))

            self.reference_img1 = pil_img
            tkImage = ImageTk.PhotoImage(pil_img)
            self.tk_img_panel_one.image = tkImage
            self.tk_img_panel_one.configure(image=tkImage)

            pil_img_dummy = Image.new('RGB', (new_img_width, new_img_height), (200, 200, 200))
            tkImage_dummy = ImageTk.PhotoImage(pil_img_dummy)
            self.reference_img2 = pil_img_dummy
            self.tk_img_panel_two.image = tkImage_dummy
            self.tk_img_panel_two.configure(image=tkImage_dummy)

            offset_y = int(0.05*self.height)
            offset_x = int(0.365*self.width) + 10       # bc line is offset for 10
            space_btwn_imgs = 10

            # "place" geometry manager has 0/0 in the upper left corner -> can easily be seen when setting x=y=0

            if img_width > img_height:      # wider than high: place above each other
                space_right_of_line = 0.635*0.5*self.width                          # get free space right of line
                img_x = offset_x + int((space_right_of_line-0.5*pil_img.size[0]))   # shift it by line offset, get half
                vertical_middle = 0.9*0.5*self.height                               # get available vertical space, middle it
                img_y1 = offset_y + vertical_middle - pil_img.size[1]
                img_y2 = offset_y + vertical_middle + space_btwn_imgs
                self.tk_img_panel_one.place(x = img_x, y = img_y1)
                self.tk_img_panel_two.place(x = img_x, y = img_y2)

            if img_height > img_width:      # higher than wide: place next to each other
                img_x1 = offset_x + int(0.635*0.5*self.width) - pil_img.size[0]       # get space right of line, divide it by two, subtract img width
                img_x2 = offset_x + int(0.635*0.5*self.width) + space_btwn_imgs       # get space right of line, add small constant
                vertical_middle = 0.9 * 0.5 * self.height  # get available vertical space, middle it
                img_y = offset_y + vertical_middle - int(pil_img.size[1]*0.5)
                self.tk_img_panel_one.place(x=img_x1, y=img_y)
                self.tk_img_panel_two.place(x=img_x2, y=img_y)

            self.print_label['text'] = "Image loaded successfully!"
            return pil_img

        else:
            self.print_label['text'] = "No valid image format. Use jpg, png or dng."
            return None


    def save_image(self):       # TODO: aktuell wird immmer nur die kleine Kopie gespeichert, wende CAN auf großes Bild an und speichere das
        if self.tk_img_panel_two.winfo_ismapped() and self.slider_variables:
            filepath = filedialog.asksaveasfilename(initialdir = os.getcwd(), title = "Save the edited image",
                                                       filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            if len(filepath.split('.')) == 1:
                filepath += '.jpg'
            self.reference_img2.save(filepath)
            self.print_label['text'] = 'Image saved sucessfully!'
        else:
            self.print_label['text'] = 'Load and edit an image first!'

    def reset_all(self):
        for slider in self.sliders:
            slider.set(0)
        self.gamma_slider.set(0.01)
        for variable in self.slider_variables:
            variable.set(0)
        #self.tk_img_panel_one.place_forget()       # leave the current image, reset only filters and edited img
        self.tk_img_panel_two.place_forget()

    def preview(self):
        # check if image is yet available, else do nothing
        if self.tk_img_panel_one.winfo_ismapped():
            current_filer_values, current_gamma = self.get_all_slider_values()
            self.nicer.set_filters(current_filer_values)
            self.nicer.set_gamma(current_gamma)
            preview_image = self.nicer.single_image_pass_can(self.reference_img1)
            self.reference_img2 = Image.fromarray(preview_image)
            tk_preview = ImageTk.PhotoImage(self.reference_img2)
            self.tk_img_panel_two.image = tk_preview
            self.tk_img_panel_two.configure(image=tk_preview)
        else:
            self.print_label['text'] = "Load image first."

    def get_all_slider_values(self):
        values = [var.get()/100.0 for var in self.slider_variables]
        print(values, self.gamma.get())
        return values, self.gamma.get()

    def nicer(self):
        print("nicer")


if __name__ == '__main__':
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    my_gui = NicerGui(root, (screen_width, screen_height))
    root.mainloop()
