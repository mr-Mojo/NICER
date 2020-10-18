#
# running_in_container = True if os.environ.get('RUNNING_IN_CONTAINER') else False
#         if running_in_container:
#             os.environ['TORCH_HOME'] = '/repo/models'
#             checkpoint_can = '/repo/NICER/' + checkpoint_can
#             checkpoint_nima = '/repo/NICER/' + checkpoint_nima

# def batch_enhance(self, folderpath):
#     self.folderpath = None
#     fileList = [x for x in folderpath if
#                 x.split('.')[-1] in config.supported_extensions or x.split('.')[-1] in config.supported_extensions_raw]
#     count = 0
#     for element in os.listdir(folderpath):
#         element_extension = element.split('.')[-1]
#         if element_extension not in config.supported_extensions_raw and element_extension not in config.supported_extensions:
#             continue
#         count += 1
#         print_msg("Enhancing image {} of {}".format(count, len(fileList)), 2)
#
#         print(element)
#         new_filename = element.replace('.' + element_extension, '_edited.' + element_extension)
#         self.reset_all()
#         # otherwise, element is either img or raw file
#         self.open_image(called_from_batch=True, img_path=os.path.join(folderpath, element))  # --> open & display
#         self.nicer_enhance(called_from_batch=True)
#         self.save_image(called_from_batch=True, save_path=os.path.join(folderpath, new_filename))
#         print("Done")


 # def open_folder(self):
 #        folderpath = filedialog.askdirectory(title="Select directory to batch-enhance")
 #        self.folderpath = folderpath
 #        if len(folderpath) is None:
 #            self.print_label['text'] = "No valid file path."
 #            return
 #        else:
 #            self.print_label['text'] = "Ready to batch-enhance folder!"
 #            #self.batch_enhance(folderpath)

#     def enhance_image_folder(self, folder_path, random=False):
#         if not os.path.exists(os.path.join(folder_path, 'results')):
#             os.mkdir(os.path.join(folder_path, 'results'))
#
#         no_of_imgs = len([x for x in os.listdir(folder_path) if x.split('.')[-1] in config.supported_extensions])
#
#         results = {}
#         for idx, img_name in enumerate(os.listdir(folder_path)):
#             img_basename = img_name.split('.')[0]
#             extension = img_name.split('.')[-1]
#             if extension not in config.supported_extensions: continue
#             print_msg("\nWorking on image {} of {}".format(idx, no_of_imgs), 1)
#
#             if random:  # make random destructive baseline
#                 random_filters = [0.0] * 8
#                 for i in range(len(random_filters)):
#                     random_filters[i] = np.random.uniform(-50,
#                                                           50) / 100.0  # filter order doesn matter, it's all random anyway
#                 self.set_filters(random_filters)
#                 results[img_name + '_init'] = self.filters.tolist()
#                 init_pil_img = Image.open(os.path.join(folder_path, img_name))
#                 init_random_img_np = self.single_image_pass_can(init_pil_img, resize=True)
#                 init_random_img_pil = Image.fromarray(init_random_img_np)
#                 init_random_img_pil.save(os.path.join(folder_path, 'results', img_basename + '_init.' + extension))
#                 enhanced_img, init_nima, final_nima = self.enhance_image(os.path.join(folder_path, img_name),
#                                                                          re_init=False)
#
#             else:
#                 enhanced_img, init_nima, final_nima = self.enhance_image(os.path.join(folder_path, img_name))
#
#             pil_img = Image.fromarray(enhanced_img)
#             pil_img.save(os.path.join(folder_path, 'results', img_basename + '_enhanced.' + extension))
#
#             results[img_name] = (init_nima, final_nima, self.filters.tolist())
#
#         json.dump(results, open(os.path.join(folder_path, 'results', "results.json"), 'w'))
#         print_msg("Saved results. Finished.", 1)
#
#
# def plot_filter_intensities(intensities_for_plot):
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#
#     from scipy.interpolate import make_interp_spline
#     x_e = np.arange(1, config.epochs + 1)
#     x = np.linspace(1, config.epochs + 1, 500).tolist()
#     sat, con, bri, sha, hig, llf, exp, nld = [], [], [], [], [], [], [], []
#
#     for key, val in intensities_for_plot.items():
#         sat.append(val[0])
#         con.append(val[1])
#         bri.append(val[2])
#         sha.append(val[3])
#         hig.append(val[4])
#         llf.append(val[5])
#         nld.append(val[6])
#         exp.append(val[7])
#
#     spl0 = make_interp_spline(x_e, sat, k=3)  # type BSpline
#     spl1 = make_interp_spline(x_e, con, k=3)  # type BSpline
#     spl2 = make_interp_spline(x_e, bri, k=3)  # type BSpline
#     spl3 = make_interp_spline(x_e, sha, k=3)  # type BSpline
#     spl4 = make_interp_spline(x_e, hig, k=3)  # type BSpline
#     spl5 = make_interp_spline(x_e, llf, k=3)  # type BSpline
#     spl6 = make_interp_spline(x_e, nld, k=3)  # type BSpline
#     spl7 = make_interp_spline(x_e, exp, k=3)  # type BSpline
#
#     a = spl0(x)
#     b = spl1(x)
#     c = spl2(x)
#     d = spl3(x)
#     e = spl4(x)
#     f = spl5(x)
#     g = spl6(x)
#     h = spl7(x)
#
#     fig, ax = plt.subplots()
#     fig.subplots_adjust(left=0.14, bottom=0.22, right=0.95, top=0.87)
#
#     h0 = ax.plot(x, a)
#     h1 = ax.plot(x, b)
#     h2 = ax.plot(x, c)
#     h3 = ax.plot(x, d)
#     h4 = ax.plot(x, e)
#     h5 = ax.plot(x, f)
#     h6 = ax.plot(x, g)
#     h7 = ax.plot(x, h)
#
#     ax.set_xlabel('Optimization Epochs')
#     ax.set_ylabel('Filter Intensity')
#
#     width = 5.487 * 2
#     height = width / 1.218
#     fig.set_size_inches(width, height)
#
#     ax.legend((h0[0], h1[0], h2[0], h3[0], h4[0], h7[0], h6[0], h5[0]),
#               ('Sat', 'Con', 'Bri', 'Sha', 'Hig', 'Exp', 'LLF', 'NLD'))
#
#     plt.show()
#
#     if config.save_filter_intensities:
#         mpl.use('pdf')
#         plt.rc('font', family='serif', serif='Times')
#         plt.rc('text', usetex=True)
#         plt.rc('xtick', labelsize=8)
#         plt.rc('ytick', labelsize=8)
#         plt.rc('axes', labelsize=8)
#         fig.savefig('results.pdf')


# import os
# import csv
# import sys
# import time
# import math
# import json
# import queue
# import torch
# import threading
# import webbrowser
#
# import argparse
# import numpy as np
# import torchvision
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from PIL import Image, ImageStat
# import torchvision.models as models
# from collections.abc import Iterable
# from torchvision.transforms import transforms
#
# import cv2
# import rawpy
# from skimage.transform import resize
# from skimage.metrics import structural_similarity as ssim
