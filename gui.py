import tkinter as tk
from tkinter import Label, Button, DoubleVar, Scale, HORIZONTAL, filedialog, IntVar, Checkbutton, Canvas
from tkinter.ttk import Label, Button

import numpy as np
import os
import queue
import threading
import torch
import webbrowser
from PIL import Image, ImageTk
from torchvision.transforms import transforms

import config
from autobright import normalize_brightness
from neural_models import CAN
from nicer import NICER, print_msg

running = True


class NicerGui:
    def __init__(self, master, screen_size):
        if screen_size[0] > 1920:
            screen_size = list(screen_size)  # when multiscreen, use only 1 screen
            screen_size[0] = 1920
            screen_size[1] = 1080
        self.master = master
        self.height = int(0.85 * screen_size[1])
        self.width = int(0.85 * screen_size[0])
        master.title("NICER - Neural Image Correction and Enhancement Routine")
        master.geometry(str(self.width) + 'x' + str(self.height))  # let gui be x% of screen, centered
        sliderlength = 200

        self.nicer = NICER(checkpoint_can=config.can_checkpoint_path, checkpoint_nima=config.nima_checkpoint_path,
                           can_arch=config.can_filter_count)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.can = CAN(no_of_filters=8) if config.can_filter_count == 8 else CAN(no_of_filters=7)
        self.can.load_state_dict(torch.load(config.can_checkpoint_path, map_location=self.device)['state_dict'])
        self.can.eval()
        self.can.to(self.device)

        # keep references to the images for further processing, bc tk.PhotoImage cannot be converted back
        self.reference_img1 = None
        self.reference_img2 = None
        self.reference_img1_fullSize = None
        self.img_namestring = None
        self.img_extension = None
        self.helper_x = None
        self.helper_y = None
        self.epochCount = None  # for display only
        self.threadKiller = threading.Event()

        # labels:
        if True:
            self.fixlabel = Label(master, text="fix?")
            self.saturation_label = Label(master, text="Saturation")
            self.contrast_label = Label(master, text="Contrast")
            self.brightness_label = Label(master, text="Brightness")
            self.shadows_label = Label(master, text="Shadows")
            self.highlights_label = Label(master, text="Highlights")
            self.exposure_label = Label(master, text="Exposure")
            self.locallaplacian_label = Label(master, text="Local Laplacian Filtering")
            self.nonlocaldehazing_label = Label(master, text="Non-Local Dehazing")
            self.gamma_label = Label(master, text="Gamma")
            self.epoch_label = Label(master, text="Epochs:")
            self.print_label = Label(master, text="Open an image to get started!")
            self.print_label.place(x=int(0.635 * self.width), y=int(0.96 * self.height))
            self.slider_labels = [self.saturation_label, self.contrast_label, self.brightness_label, self.shadows_label,
                                  self.highlights_label, self.exposure_label, self.locallaplacian_label,
                                  self.nonlocaldehazing_label]

        # sliders:
        if True:
            self.saturation = DoubleVar()
            self.saturation_isfixed = IntVar()
            self.saturation_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                           var=self.saturation)
            self.saturation_checkbox = Checkbutton(master, var=self.saturation_isfixed)

            self.contrast = DoubleVar()
            self.contrast_isfixed = IntVar()
            self.contrast_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                         var=self.contrast)
            self.contrast_checkbox = Checkbutton(master, var=self.contrast_isfixed)

            self.brightness = DoubleVar()
            self.brightess_isfixed = IntVar()
            self.brightness_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                           var=self.brightness)
            self.brigtness_checkbox = Checkbutton(master, var=self.brightess_isfixed)

            self.shadows = DoubleVar()
            self.shadows_isfixed = IntVar()
            self.shadows_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                        var=self.shadows)
            self.shadows_checkbox = Checkbutton(master, var=self.shadows_isfixed)

            self.highlights = DoubleVar()
            self.highlighs_isfixed = IntVar()
            self.highlights_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                           var=self.highlights)
            self.highlights_checkbox = Checkbutton(master, var=self.highlighs_isfixed)

            self.exposure = DoubleVar()
            self.exposure_isfixed = IntVar()
            self.exposure_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                         var=self.exposure)
            self.exposure_checkbox = Checkbutton(master, var=self.exposure_isfixed)

            self.locallaplacian = DoubleVar()
            self.locallaplacian_isfixed = IntVar()
            self.locallaplacian_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                               var=self.locallaplacian)
            self.locallaplacian_checkbox = Checkbutton(master, var=self.locallaplacian_isfixed)

            self.nonlocaldehazing = DoubleVar()
            self.nonlocaldehazing_isfixed = IntVar()
            self.nonlocaldehazing_slider = Scale(master, from_=-100, to=100, length=sliderlength, orient=HORIZONTAL,
                                                 var=self.nonlocaldehazing)
            self.nonlocaldehazing_checkbox = Checkbutton(master, var=self.nonlocaldehazing_isfixed)

            self.gamma = DoubleVar()
            self.gamma_slider = Scale(master, from_=0.005, to=0.5, length=sliderlength, orient=HORIZONTAL,
                                      var=self.gamma, resolution=0.005)
            self.gamma_slider.set(0.1)

            self.slider_variables = [self.saturation, self.contrast, self.brightness, self.shadows,
                                     self.highlights, self.exposure, self.locallaplacian, self.nonlocaldehazing]
            self.checkbox_variables = [self.saturation_isfixed, self.contrast_isfixed, self.brightess_isfixed,
                                       self.shadows_isfixed, self.highlighs_isfixed, self.exposure_isfixed,
                                       self.locallaplacian_isfixed, self.nonlocaldehazing_isfixed]
            self.checkboxes = [self.saturation_checkbox, self.contrast_checkbox, self.brigtness_checkbox,
                               self.shadows_checkbox, self.highlights_checkbox, self.exposure_checkbox,
                               self.locallaplacian_checkbox, self.nonlocaldehazing_checkbox]
            self.sliders = [self.saturation_slider, self.contrast_slider, self.brightness_slider, self.shadows_slider,
                            self.highlights_slider, self.exposure_slider, self.locallaplacian_slider,
                            self.nonlocaldehazing_slider]

        # create images and line, and place them
        if True:
            w = Canvas(master, width=20, height=self.height)
            w.place(x=int(0.365 * self.width), y=10)
            w.create_line(10, 20, 10, int(0.9 * self.height), fill="#476042", dash=(4, 4))

            pil_img_one = Image.new('RGB', (224, 224), (255, 255, 255))
            pil_img_two = Image.new('RGB', (224, 224), (150, 150, 150))
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
            three_quarters = int(0.65 * self.height)
            space = (three_quarters - 30) / 7

            for idx, label in enumerate(self.slider_labels):
                label.place(x=20, y=30 + idx * space)
            for idx, slider in enumerate(self.sliders):
                slider.place(x=150, y=10 + idx * space)
            for idx, chckbx in enumerate(self.checkboxes):
                chckbx.place(x=150 + sliderlength + 10, y=30 + idx * space)

            self.fixlabel.place(x=360, y=10)
            gamma_space = 2 * space if space < 60 else 120
            self.gamma_label.place(x=20, y=50 + 6 * space + gamma_space)
            self.gamma_slider.place(x=150, y=30 + 6 * space + gamma_space)

        # create buttons and place
        if True:
            self.open_button = Button(master, text="Open Image", command=self.open_image)
            self.save_button = Button(master, text="Save Image", command=self.save_image)
            self.reset_button = Button(master, text="Reset", command=self.reset_all)
            self.preview_button = Button(master, text="Preview", command=self.preview)
            self.nicer_button = Button(master, text="NICER!", command=self.nicer_routine)
            self.stop_button = Button(master, text="Stop!", command=self.stop)
            self.about_button = Button(master, text="About", command=self.about)

            button_y = 50 + 6 * space + gamma_space + 50
            self.open_button.place(x=40, y=button_y + 20)
            self.save_button.place(x=40, y=button_y + 60)
            self.nicer_button.place(x=40 + 105, y=button_y + 20)
            self.stop_button.place(x=40 + 105, y=button_y + 60)
            self.preview_button.place(x=40 + 200, y=button_y + 20)
            self.reset_button.place(x=40 + 200, y=button_y + 60)
            self.about_button.place(x=40 + 105, y=button_y + 100)

            self.epoch_label.place(x=40 + 105, y=button_y - 5)
            self.epochVar = IntVar()
            self.epochBox = tk.Entry(master, width=3)
            self.epochBox.insert(-1, '50')
            self.epochBox.place(x=40 + 160, y=button_y - 5)

    def about(self):
        url = "https://github.com/mr-Mojo/NICER#how-does-it-work"  # TODO: nach push ändern
        webbrowser.open(url, new=1)

    def stop(self):
        global running
        running = False

    def open_image(self, img_path=None):
        """
        opens an image, if the image extension is supported in config.py. Currently supported extensions are jpg, png and dng, although
        more might work. The image is resized for displaying. A reference for further processing is stored in self.reference_img_fullSize.
        """

        filepath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select an image to open",
                                              filetypes=(
                                                  ("jpg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
        if filepath is None: return

        if filepath.split('.')[-1] in config.supported_extensions:

            self.img_namestring = filepath.split('.')[0]
            self.img_extension = filepath.split('.')[-1]

            pil_img = Image.open(filepath)

            img_width = pil_img.size[0]
            img_height = pil_img.size[1]

            # dirty hack to avoid errors when image is a square:
            if img_width == img_height:
                pil_img = pil_img.resize((img_width, img_height - 1))
                img_width = pil_img.size[0]
                img_height = pil_img.size[1]

            self.reference_img1_fullSize = pil_img

            print_msg("Full image size: {}x{}".format(img_width, img_height), 3)

            # resize images so that they can fit next to each other:

            # margins of the display canvas: 0.6*windowwidth, 0.9*windowheight
            if img_width > img_height:
                max_width = int(0.6 * self.width)  # images wider than high: place above each other
                max_height = int(0.5 * 0.9 * self.height)  # max height is half the display canvas

            elif img_height > img_width:
                max_width = int(0.5 * 0.6 * self.width)  # images higher than wide: place next to each other
                max_height = int(0.9 * self.width)

            max_size = int(0.9 * self.height) if img_height > img_width else int(0.6 * self.width)

            # just to that there are no errors if image is not resized:
            new_img_width = img_width
            new_img_height = img_height

            # img too large: either exceeds max size, or it is too broad / high to be displayed next to each other
            if max(pil_img.size) > max_size or pil_img.size[1] > max_height or pil_img.size[0] > max_width:
                longer_side = max(pil_img.size)
                factor = max_size / longer_side
                new_img_width = int(img_width * factor)
                new_img_height = int(img_height * factor)

                if img_width > img_height and new_img_height > max_height:  # landscape format
                    while new_img_height > max_height:
                        factor *= 0.99
                        print_msg("reduced factor to %f" % factor, 3)
                        new_img_width = int(img_width * factor)
                        new_img_height = int(img_height * factor)

                elif img_height > img_width and new_img_width > max_width:  # portrait format
                    while new_img_width > max_width:
                        factor *= 0.99
                        print_msg("reduced factor to %f" % factor, 3)
                        new_img_width = int(img_width * factor)
                        new_img_height = int(img_height * factor)

                # img is now resized in a way for 2 images to fit next to each other
                pil_img = pil_img.resize((new_img_width, new_img_height))

            self.reference_img1 = pil_img
            tkImage = ImageTk.PhotoImage(pil_img)
            self.tk_img_panel_one.image = tkImage
            self.tk_img_panel_one.configure(image=tkImage)
            print_msg("Display image size: {}x{}".format(new_img_width, new_img_height), 3)

            pil_img_dummy = Image.new('RGB', (new_img_width, new_img_height), (200, 200, 200))
            tkImage_dummy = ImageTk.PhotoImage(pil_img_dummy)
            self.reference_img2 = pil_img_dummy
            self.tk_img_panel_two.image = tkImage_dummy
            self.tk_img_panel_two.configure(image=tkImage_dummy)

            offset_y = int(0.05 * self.height)
            offset_x = int(0.365 * self.width) + 10  # bc line is offset for 10
            space_btwn_imgs = 10

            # "place" geometry manager has 0/0 in the upper left corner

            if img_width > img_height:  # wider than high: place above each other
                space_right_of_line = 0.635 * 0.5 * self.width  # get free space right of line
                img_x = offset_x + int(
                    (space_right_of_line - 0.5 * pil_img.size[0]))  # shift it by line offset, get half
                vertical_middle = 0.9 * 0.5 * self.height  # get available vertical space, middle it
                img_y1 = offset_y + vertical_middle - pil_img.size[1]
                img_y2 = offset_y + vertical_middle + space_btwn_imgs
                self.tk_img_panel_one.place(x=img_x, y=img_y1)
                self.tk_img_panel_two.place(x=img_x, y=img_y2)
                self.helper_x = img_x
                self.helper_y = img_y2

            if img_height > img_width:  # higher than wide: place next to each other
                img_x1 = offset_x + int(0.635 * 0.5 * self.width) - pil_img.size[
                    0]  # get space right of line, divide it by two, subtract img width
                img_x2 = offset_x + int(
                    0.635 * 0.5 * self.width) + space_btwn_imgs  # get space right of line, add small constant
                vertical_middle = 0.9 * 0.5 * self.height  # get available vertical space, middle it
                img_y = offset_y + vertical_middle - int(pil_img.size[1] * 0.5)
                self.tk_img_panel_one.place(x=img_x1, y=img_y)
                self.tk_img_panel_two.place(x=img_x2, y=img_y)
                self.helper_x = img_x2
                self.helper_y = img_y

            self.print_label['text'] = "Image loaded successfully!"
            return pil_img

        else:
            self.print_label['text'] = "No valid image format. Use a format specified in the config."
            return None

    # final can pass with found filter intensities happens in save_image for the full resolution image
    # (to save time during optimization, we optimize on the rescaled image)
    def save_image(self, save_path=None):
        """
        saves an image, if it has previously been modified (i.e., if the slider values != 0).
        The unedited full size reference image is stored in self.reference_img_1_fullSize.
        """

        if self.tk_img_panel_two.winfo_ismapped() and self.slider_variables:
            filepath = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save the edited image",
                                                    filetypes=(("as jpg file", "*.jpg"),
                                                               # ("as raw file", "*.png"),
                                                               ("all files", "*.*")))

            if len(filepath.split('.')) == 1:
                filepath += '.jpg'

            self.print_label['text'] = 'Saving image...'

        else:
            self.print_label['text'] = 'Load and edit an image first!'
            return

        current_filer_values, current_gamma = self.get_all_slider_values()
        self.nicer.set_filters(current_filer_values)
        self.nicer.set_gamma(current_gamma)

        # calc a factor for resizing to config.final_size
        width, height = self.reference_img1_fullSize.size  # img.size: (width, height)

        if width > config.final_size or height > config.final_size:
            print("resize")
            print_msg("Resizing to {}p before saving".format(str(config.final_size)), 3)
            factor = config.final_size / height if (height > width) else config.final_size / width

            width = int(width * factor)
            height = int(height * factor)

            try:
                hd_image = self.nicer.single_image_pass_can(self.reference_img1_fullSize.resize((width, height)),
                                                            abn=True)
            except RuntimeError:
                # image and model too large for GPU
                hd_image = self.nicer.single_image_pass_can(self.reference_img1_fullSize.resize((width, height)),
                                                            abn=True, mapToCpu=True)


        else:
            # dims < config.final_size on the longest side, no resizing
            try:
                hd_image = self.nicer.single_image_pass_can(self.reference_img1, abn=True)
            except RuntimeError:
                hd_image = self.nicer.single_image_pass_can(self.reference_img1, abn=True, mapToCpu=True)

        if hd_image is not None:
            hd_image_pil = Image.fromarray(hd_image)
            hd_image_pil.save(filepath)
            self.print_label['text'] = 'Image saved successfully!'
        else:
            self.print_label['text'] = 'Could not save image. See console output.'

    def reset_all(self):
        """ reset GUI and slider values """
        for slider in self.sliders:
            slider.set(0)
        self.gamma_slider.set(0.100)
        for variable in self.slider_variables:
            variable.set(0)
        for checkbox in self.checkboxes:
            checkbox.deselect()
        for variable in self.checkbox_variables:
            variable.set(0)

        self.epochBox.delete(0, 'end')
        self.epochBox.insert(-1, '50')
        self.threadKiller.clear()

        # self.tk_img_panel_one.place_forget()       # leave the current image, reset only filters and edited img
        self.tk_img_panel_two.place_forget()

    def preview(self, filterList=None):
        """ apply the currently set slider combination onto the image (using resized img for increased speed) """
        # check if image is yet available, else do nothing
        if not self.tk_img_panel_one.winfo_ismapped():
            self.print_label['text'] = "Load image first."
            return

        current_filter_values, current_gamma = self.get_all_slider_values()

        self.nicer.set_gamma(current_gamma)
        self.nicer.set_filters(current_filter_values)
        preview_image = self.nicer.single_image_pass_can(self.reference_img1, abn=True)
        self.reference_img2 = Image.fromarray(preview_image)
        self.display_img_two()

    def display_img_two(self):
        tk_preview = ImageTk.PhotoImage(self.reference_img2)
        self.tk_img_panel_two.place(x=self.helper_x, y=self.helper_y)
        self.tk_img_panel_two.image = tk_preview
        self.tk_img_panel_two.configure(image=tk_preview)

    def get_all_slider_values(self):
        values = [var.get() / 100.0 for var in self.slider_variables]
        print_msg("Sliders values: {} -- Gamma: {}".format(values, self.gamma.get()), 3)
        return values, self.gamma.get()

    def get_all_checkbox_values(self):
        values = [var.get() for var in self.checkbox_variables]
        print_msg("Fixed Filters: {}".format(values), 3)
        gui_exp = values[5]
        gui_llf = values[6]
        gui_nld = values[7]
        values[5] = gui_llf
        values[6] = gui_nld
        values[7] = gui_exp  # change indices bc exp in CAN is #8 but in GUI is #5
        return values

    def set_all_image_filter_sliders(self, valueList):
        # does not set gamma. called by nicer_enhance routine to display final outcome of enhancement
        for i in range(5):
            self.sliders[i].set(valueList[i])
            self.slider_variables[i].set(valueList[i])

        # valueList is returned by NICER and has format sat-con-bri-sha-hig-llf-nld-exp
        # sliders is controlled by gui and has format sat-con-bri-sha-hig-exp-llf-nld
        self.sliders[6].set(valueList[5])
        self.sliders[7].set(valueList[6])
        self.sliders[5].set(valueList[7])
        self.slider_variables[6].set(valueList[5])
        self.slider_variables[7].set(valueList[6])
        self.slider_variables[5].set(valueList[7])

    def checkqueue(self):
        while self.nicer.queue.qsize():
            try:
                msg = self.nicer.queue.get(0)
                if isinstance(msg, int):  # passed the epoch count
                    self.print_label['text'] = 'Optimizing epoch {} of {}'.format(str(msg), self.epochCount)
                elif isinstance(msg, list):  # passed the filter values
                    self.set_all_image_filter_sliders([x * 100 for x in msg])
                elif isinstance(msg, np.ndarray):  # thread terminated, passed the last enhanced image
                    enhanced_img_pil = Image.fromarray(msg)
                    self.reference_img2 = enhanced_img_pil
                    self.display_img_two()

                else:  # passed 'dummy' string to get here

                    if not config.preview: return
                    # previewing while optimization, need to do filter setting and can pass manually
                    # cannot use enhanced img from can optimization, as its wrong format (224x224)
                    current_filter_values, gamma = self.get_all_slider_values()
                    filterValues = [0.0] * 8
                    for i in range(5):
                        filterValues[i] = current_filter_values[i]
                    filterValues[5] = current_filter_values[6]  # llf is 5 in can but 6 in gui (bc exp is inserted)
                    filterValues[6] = current_filter_values[7]  # nld is 6 in can but 7 in gui
                    filterValues[7] = current_filter_values[5]  # exp is 7 in can but 5 in gui

                    preview_image = self.alternate_can(self.reference_img1, filterValues)
                    self.reference_img2 = Image.fromarray(preview_image)
                    self.display_img_two()

            except queue.Queue.Empty:
                pass

    def periodiccall(self):
        self.checkqueue()
        if self.thread.is_alive() and running:  # running is set to false by stop button
            self.master.after(100, self.periodiccall)
        elif not self.thread.is_alive() and running:  # thread terminated naturally, after optimization
            self.print_label['text'] = "Optimization finished."
            self.nicer_button.config(state="active")
            self.nicer.queue = queue.Queue()
        else:
            self.threadKiller.set()  # thread killed by stop button
            self.thread.join()
            self.print_label['text'] = "Stopped optimization."
            self.nicer_button.config(state="active")
            self.threadKiller.clear()
            self.nicer.queue = queue.Queue()

    def nicer_routine(self):
        global running
        running = True
        self.nicer_button.config(state="disabled")

        self.nicer_enhance()

    def nicer_enhance(self):

        # check if image is yet available, else do nothing
        if self.tk_img_panel_one.winfo_ismapped():

            self.nicer.re_init()  # reset everything, especially optimizers, for a fresh optimization

            # get slider values and set them for the optimization routine
            slider_vals, gamma = self.get_all_slider_values()
            checkbox_vals = self.get_all_checkbox_values()
            self.nicer.set_filters(slider_vals)
            self.nicer.set_gamma(gamma)

            custom_filters = False
            for value in slider_vals:
                if value != 0.0: custom_filters = True

            self.epochCount = int(self.epochBox.get())

            if not custom_filters:
                print_msg("All filters are zero.", 2)
                self.thread = threading.Thread(target=self.nicer.enhance_image, args=(self.reference_img1,
                                                                                      True, checkbox_vals,
                                                                                      self.epochCount,
                                                                                      self.threadKiller), daemon=True)
            else:
                print_msg("Using user-defined filter preset", 2)
                self.thread = threading.Thread(target=self.nicer.enhance_image, args=(self.reference_img1,
                                                                                      False, checkbox_vals,
                                                                                      self.epochCount,
                                                                                      self.threadKiller), daemon=True)

            self.thread.start()
            self.periodiccall()

        else:
            self.print_label['text'] = "Load image first."

    def alternate_can(self, image, filterList):
        # alternate CAN for previewing the images, as NICER's CAN is used for gradient computation
        # filterList is passable since while optimizing, we cannot use nicer.filters, as these hold the gradients

        # gets called from periodiccall queue handler
        bright_norm_img = normalize_brightness(image, input_is_PIL=True)
        image = Image.fromarray(bright_norm_img)
        image_tensor = transforms.ToTensor()(image)

        filter_tensor = torch.zeros((8, image_tensor.shape[1], image_tensor.shape[2]),
                                    dtype=torch.float32).to(self.device)  # tensorshape [c,w,h]
        for l in range(8):
            filter_tensor[l, :, :] = filterList[l]  # construct uniform filtermap
        mapped_img = torch.cat((image_tensor.cpu(), filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(self.device)

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')

        # returns a np.array of type np.uint8
        return enhanced_clipped
