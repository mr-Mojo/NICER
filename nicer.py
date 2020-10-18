import queue
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms

import config
from autobright import normalize_brightness
from neural_models import error_callback, CAN, NIMA_VGG
from utils import nima_transform, print_msg, loss_with_l2_regularization


class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device='cpu', can_arch=8):
        super(NICER, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using", self.device)

        if can_arch != 8 and can_arch != 7:
            error_callback('can_arch')

        can = CAN(no_of_filters=8) if can_arch == 8 else CAN(no_of_filters=7)
        can.load_state_dict(torch.load(checkpoint_can, map_location=self.device)['state_dict'])
        can.eval()
        can.to(self.device)

        nima = NIMA_VGG(models.vgg16(pretrained=True))
        nima.load_state_dict(torch.load(checkpoint_nima, map_location=self.device))
        nima.eval()
        nima.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        self.queue = queue.Queue()

        # self.filters is a leaf-variable, bc it's created directly and not as part of an operation
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True,
                                    device=self.device)
        self.can = can
        self.nima = nima

        self.gamma = config.gamma

        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def forward(self, image, fixedFilters=None):
        filter_tensor = torch.zeros((8, 224, 224), dtype=torch.float32).to(self.device)

        # construct filtermap uniformly from given filters
        for l in range(8):
            if fixedFilters:  # filters were fixed in GUI, always use their passed values
                if fixedFilters[l][0] == 1:
                    filter_tensor[l, :, :] = fixedFilters[l][1]
                else:
                    filter_tensor[l, :, :] = self.filters.view(-1)[l]
            else:
                filter_tensor[l, :, :] = self.filters.view(-1)[l]

        mapped_img = torch.cat((image, filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(
            self.device)  # concat filters and img
        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        distr_of_ratings = self.nima(enhanced_img)  # get nima score distribution -> tensor

        self.queue.put('dummy')  # dummy

        return distr_of_ratings, enhanced_img

    def set_filters(self, filter_list):
        # usually called from GUI
        if max(filter_list) > 1:
            filter_list = [x / 100.0 for x in filter_list]

        with torch.no_grad():
            for i in range(5):
                self.filters[i] = filter_list[i]
            self.filters[5] = filter_list[6]  # llf is 5 in can but 6 in gui (bc exp is inserted)
            self.filters[6] = filter_list[7]  # nld is 6 in can but 7 in gui
            self.filters[7] = filter_list[5]  # exp is 7 in can but 5 in gui

    def set_gamma(self, gamma):
        self.gamma = gamma

    def single_image_pass_can(self, image, abn=False, filterList=None, mapToCpu=False):
        """
            pass an image through the CAN architecture 1 time. This is usually called from the GUI, to preview the images.
            It is also called when the image is to be saved, since we then need to apply the final filter intensities onto the image.

            if called_to_save_raw is False, this method will return an 8bit image to show what the current filter combination looks
            like (because PIL cannot process 16bit). If called_to_save_raw is true, it will return the enhanced 16bit image as
            np.uint16 array, to be saved with opencv.imwrite() as 16bit png.
        """

        # filterList is passable as argument because for previwing the imgs in the GUI while optimizing,
        # we cannot use self.filters, as this is used for gradient computatation

        device = self.device if mapToCpu is False else 'cpu'

        if abn:
            bright_norm_img = normalize_brightness(image, input_is_PIL=True)
            image = Image.fromarray(bright_norm_img)

        if image.size[1] > config.final_size or image.size[0] > config.final_size:
            image_tensor = transforms.Compose([
                transforms.Resize(config.final_size),
                transforms.ToTensor()])(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        filter_tensor = torch.zeros((8, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32).to(
            device)  # tensorshape [c,w,h]
        for l in range(8):
            filter_tensor[l, :, :] = filterList[l] if filterList else self.filters.view(-1)[l]

        mapped_img = torch.cat((image_tensor.cpu(), filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(device)

        try:
            enhanced_img = self.can(mapped_img)  # enhance img with CAN
        except RuntimeError:
            self.can.to('cpu')
            try:
                enhanced_img = self.can(mapped_img)  # enhance img with CAN
            except RuntimeError:
                print("DefaultCPUAllocator - not enough memory to perform this operation")
                return None
            self.can.to('cuda')

        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')

        # returns a np.array of type np.uint8

        return enhanced_clipped

    def re_init(self):
        # deprecated, formerly used for batch mode
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True,
                                    device=self.device)
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def enhance_image(self, image_path, re_init=True, fixFilters=None, epochs=config.epochs, thread_stopEvent=None):
        """
            optimization routine that is called to enhance an image.
            Usually this is called from the NICER button in the GUI.
            Accepts image path as a string, but also as PIL image.

            Returns a re-sized 8bit image as np.array
        """

        if re_init:
            self.re_init()
        else:
            # re-init is false, i.e. use user_preset filters that are selected in the GUI
            # re-init can be seen as test whether initial filter values (!= 0) should be used or not during optimization
            user_preset_filters = [self.filters[x].item() for x in range(8)]

        if isinstance(image_path, str):
            bright_normalized_img = normalize_brightness(image_path)
            pil_image = Image.fromarray(bright_normalized_img)
        else:
            pil_image = image_path
            bright_normalized_img = normalize_brightness(pil_image, input_is_PIL=True)
            pil_image = Image.fromarray(bright_normalized_img)

        image_tensor_transformed = nima_transform(pil_image)

        if fixFilters:  # fixFilters is bool list of filters to be fixed
            initial_filter_values = []
            for k in range(8):
                if fixFilters[k] == 1:
                    initial_filter_values.append([1, self.filters[k].item()])
                else:
                    initial_filter_values.append([0, self.filters[k].item()])

        # optimize image:
        print_msg("Starting optimization", 2)
        start_time = time.time()
        for i in range(epochs):
            if thread_stopEvent.is_set(): break

            print_msg("Iteration {} of {}".format(i, epochs), 2)

            if fixFilters:
                distribution, enhanced_img = self.forward(image_tensor_transformed, fixedFilters=initial_filter_values)
            else:
                distribution, enhanced_img = self.forward(image_tensor_transformed)

            self.optimizer.zero_grad()

            if re_init:
                # new for each image
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(), gamma=self.gamma)
            else:
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(),
                                                   initial_filters=user_preset_filters, gamma=self.gamma)

            loss.backward()
            self.optimizer.step()

            filters_for_queue = [self.filters[x].item() for x in range(8)]
            self.queue.put(i + 1)
            self.queue.put(filters_for_queue)

        if not thread_stopEvent.is_set():
            print_msg("Optimization for %d epochs took %.3fs" % (epochs, time.time() - start_time), 2)

            # the entire rescale thing is not needed, bc optimization happens on a smaller image (for speed improvement)
            # real rescale is done during saving.
            original_tensor_transformed = transforms.ToTensor()(pil_image)

            final_filters = torch.zeros((8, original_tensor_transformed.shape[1], original_tensor_transformed.shape[2]),
                                        dtype=torch.float32).to(self.device)
            for k in range(8):
                if fixFilters:
                    if fixFilters[k] == 1:
                        final_filters[k, :, :] = initial_filter_values[k][1]
                    else:
                        final_filters[k, :, :] = self.filters.view(-1)[k]
                else:
                    final_filters[k, :, :] = self.filters.view(-1)[k]

            strings = ['Sat', 'Con', 'Bri', 'Sha', 'Hig', 'LLF', 'NLD', 'EXP']
            print_msg("Final Filter Intensities: {}".format(
                [strings[k] + ": " + str(final_filters[k, 0, 0].item() * 100) for k in range(8)]), 3)
            self.queue.put([final_filters[k, 0, 0].item() for k in range(8)])

            mapped_img = torch.cat((original_tensor_transformed, final_filters.cpu()), dim=0).unsqueeze(dim=0).to(
                self.device)
            enhanced_img = self.can(mapped_img)
            enhanced_img = enhanced_img.cpu()
            enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()
            enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
            enhanced_clipped = enhanced_clipped.astype('uint8')

            self.queue.put(enhanced_clipped)

            # returns an 8bit image in any case ---
            return enhanced_clipped, None, None
