from imports import *
from utils import *
from neural_models import *

class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device='cpu', can_arch=8):
        super(NICER, self).__init__()
        self.device = device

        if can_arch != 8 and can_arch != 7:
            error_callback('can_arch')

        can = CAN(no_of_filters=8) if can_arch == 8 else CAN(no_of_filters=7)
        can.load_state_dict(torch.load(checkpoint_can, map_location=device)['state_dict'])
        can.eval()
        can.to(device)

        nima = NIMA_VGG(models.vgg16(pretrained=True))
        nima.load_state_dict(torch.load(checkpoint_nima, map_location=device))
        nima.eval()
        nima.to(device)

        # self.filters is a "leaf-variable", bc it's created directly and not as part of an operation
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True, device=device)
        self.can = can
        self.nima = nima
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')
        self.gamma = config.gamma

    def forward(self, image):
        filter_tensor = torch.zeros((8, 224, 224), dtype=torch.float32).to(self.device)
        for l in range(8):
            filter_tensor[l, :, :] = self.filters.view(-1)[l]  # construct filtermap uniformly from given filters
        mapped_img = torch.cat((image, filter_tensor), dim=0).unsqueeze(dim=0)  # concat filters and img

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        distr_of_ratings = self.nima(enhanced_img)  # get nima score distribution -> tensor

        return distr_of_ratings, enhanced_img

    def set_filters(self, filter_list):
        with torch.no_grad():
            for i in range(5):
                self.filters[i] = filter_list[i]
            self.filters[7] = filter_list[5]    # exposure is in 5 filterlist but 7 in can
            self.filters[5] = filter_list[6]    # llf is 6 in filterlist but 5 in can
            self.filters[6] = filter_list[7]    # nld is 7 in filterlist but 6 in can

    def set_gamma(self, gamma):
        config.gamma = gamma
        self.gamma = gamma

    def single_image_pass_can(self, image):
        image_tensor = transforms.ToTensor()(image)        # gets called from gui, with a non-tensor image
        filter_tensor = torch.zeros((8, image.size[1], image.size[0]), dtype=torch.float32).to(self.device)
        for l in range(8):
            filter_tensor[l, :, :] = self.filters.view(-1)[l]  # construct filtermap uniformly from given filters
        mapped_img = torch.cat((image_tensor, filter_tensor), dim=0).unsqueeze(dim=0)  # concat filters and img

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')
        return enhanced_clipped

    def re_init(self):
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True, device=self.device)
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    # returns enhanced image as np array
    def enhance_image(self, image_path, re_init=True, rescale_to_hd=True, verbose=False):        # accepts image_path as string, but also as PIL image object

        if re_init:
            self.re_init()      # not called from NICER button, but from batch mode in optimize_whole_folder -> new filters for each image run
        else:
            user_preset_filters = [self.filters[x].item() for x in range(8)]
        # re-init can be seen as test whether initial filter values (!= 0) should be used or not during optimization

        if isinstance(image_path, str):
            pil_image = load_pil_img(image_path)
        else:
            pil_image = image_path

        print("test")
        image_tensor_transformed = nima_transform(pil_image)
        image_tensor_transformed_batched = image_tensor_transformed.unsqueeze(dim=0)

        with torch.no_grad():
            initial_nima_score = get_tensor_mean_as_float(self.nima(image_tensor_transformed_batched))
            first_pass_distribution, _ = self.forward(image_tensor_transformed)
            initial_nima_score_after_first_pass = get_tensor_mean_as_float(first_pass_distribution)

        nima_offset = initial_nima_score - initial_nima_score_after_first_pass
        epochs = config.epochs
        losses = []

        filters_plot = {}

        # optimize image:
        print("Starting optimization")
        start_time = time.time()
        for i in range(epochs):
            if verbose:
                print("Iteration {} of {}".format(i, epochs))
            distribution, enhanced_img = self.forward(image_tensor_transformed)
            self.optimizer.zero_grad()
            if re_init:
                filters_plot[i] = [self.filters[x].item() for x in range(8)]
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu())      # re-init True, i.e. new for each image
                losses.append(loss.item())
            else:
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(), initial_filters=user_preset_filters)       # TODO: test, and gamma too
                losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        print("Optimization for %d epochs took %.3fs" % (epochs, time.time() - start_time))

        import matplotlib.pyplot as plt
        from scipy.interpolate import make_interp_spline, BSpline

        x_e = np.arange(1,config.epochs+1)
        x = np.linspace(1,config.epochs+1,500).tolist()
        sat, con, bri, sha, hig, llf, exp, nld = [], [], [], [], [], [], [], []

        for key, val in filters_plot.items():
            sat.append(val[0])
            con.append(val[1])
            bri.append(val[2])
            sha.append(val[3])
            hig.append(val[4])
            llf.append(val[5])
            nld.append(val[6])
            exp.append(val[7])

        spl0 = make_interp_spline(x_e, sat, k=3)  # type BSpline
        spl1 = make_interp_spline(x_e, con, k=3)  # type BSpline
        spl2 = make_interp_spline(x_e, bri, k=3)  # type BSpline
        spl3 = make_interp_spline(x_e, sha, k=3)  # type BSpline
        spl4 = make_interp_spline(x_e, hig, k=3)  # type BSpline
        spl5 = make_interp_spline(x_e, llf, k=3)  # type BSpline
        spl6 = make_interp_spline(x_e, nld, k=3)  # type BSpline
        spl7 = make_interp_spline(x_e, exp, k=3)  # type BSpline

        a = spl0(x)
        b = spl1(x)
        c = spl2(x)
        d = spl3(x)
        e = spl4(x)
        f = spl5(x)
        g = spl6(x)
        h = spl7(x)

        h0 = plt.plot(x, a)
        h1 = plt.plot(x, b)
        h2 = plt.plot(x, c)
        h3 = plt.plot(x, d)
        h4 = plt.plot(x, e)
        h5 = plt.plot(x, f)
        h6 = plt.plot(x, g)
        h7 = plt.plot(x, h)

        plt.legend((h0[0], h1[0], h2[0], h3[0],h4[0],h5[0],h6[0],h7[0]), ('sat','con','bri','sha','hig','llf','nld','exp'))
        plt.show() # debug

        if rescale_to_hd:
            if pil_image.size[0] > config.final_size or pil_image.size[1] > config.final_size:
                original_tensor_transformed = hd_transform(pil_image)
            else:
                original_tensor_transformed = transforms.ToTensor()(pil_image)
        else:
            original_tensor_transformed = transforms.ToTensor()(pil_image)

        final_filters = torch.zeros((8, original_tensor_transformed.shape[1], original_tensor_transformed.shape[2]), dtype=torch.float32).to(self.device)
        for k in range(8):
            final_filters[k, :, :] = self.filters.view(-1)[k]

        mapped_img = torch.cat((original_tensor_transformed, final_filters.cpu()), dim=0).unsqueeze(dim=0).to(self.device)
        enhanced_img = self.can(mapped_img)

        retransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        nima_sized_img = retransform(enhanced_img.squeeze(dim=0).cpu()).to(self.device)

        with torch.no_grad():
            prediction = self.nima(nima_sized_img.unsqueeze(dim=0))
            final_nima_score = get_tensor_mean_as_float(prediction) + nima_offset

        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')

        return enhanced_clipped, initial_nima_score, final_nima_score

    def enhance_image_folder(self, folder_path):
        if not os.path.exists(os.path.join(folder_path, 'results')):
            os.mkdir(os.path.join(folder_path, 'results'))

        no_of_imgs = len([x for x in os.listdir(folder_path) if x.split('.')[-1] in config.supported_extensions])

        results = {}
        for idx, img_name in enumerate(os.listdir(folder_path)):
            extension = img_name.split('.')[-1]
            if extension not in config.supported_extensions: continue
            print("Working on image {} of {}".format(idx, len(os.listdir(folder_path))))

            enhanced_img, init_nima, final_nima = self.enhance_image(os.path.join(folder_path, img_name))

            img_basename = img_name.split('.')[0]
            pil_img = Image.fromarray(enhanced_img)
            pil_img.save(os.path.join(folder_path, 'results', img_basename + '_enhanced.' + extension))

            results[img_name] = (init_nima, final_nima, self.filters.tolist())

        json.dump(results, open(os.path.join(folder_path, 'results', "results.json"), 'w'))
        print("Saved results. Finished.")
