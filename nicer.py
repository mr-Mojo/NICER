from imports import *
from utils import *
from neural_models import *
from autobright import normalize_brightness

class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device='cpu', can_arch=8):
        super(NICER, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        running_in_container = True if os.environ.get('RUNNING_IN_CONTAINER') else False
        if running_in_container:
            os.environ['TORCH_HOME'] = '/repo/models'
            checkpoint_can = '/repo/NICER/' + checkpoint_can
            checkpoint_nima = '/repo/NICER/' + checkpoint_nima

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
        self.gamma = config.gamma

        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    def forward(self, image):
        filter_tensor = torch.zeros((8, 224, 224), dtype=torch.float32).to(self.device)
        for l in range(8):
            filter_tensor[l, :, :] = self.filters.view(-1)[l]  # construct filtermap uniformly from given filters
        mapped_img = torch.cat((image, filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(self.device)  # concat filters and img

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        distr_of_ratings = self.nima(enhanced_img)  # get nima score distribution -> tensor

        return distr_of_ratings, enhanced_img

    def set_filters(self, filter_list):
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

    def single_image_pass_can(self, image, resize=True, abn=False):

        if abn:
            bright_norm_img = normalize_brightness(image, input_is_PIL=True)
            image = Image.fromarray(bright_norm_img)

        if image.size[1] > config.final_size or image.size[0] > config.final_size:
            image_tensor = transforms.Compose([
                transforms.Resize(config.final_size),
                transforms.ToTensor()])(image)
        else:
            image_tensor = transforms.ToTensor()(image)  # gets called from gui, with a non-tensor image


        filter_tensor = torch.zeros((8, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32).to(self.device)     # tensorshape [c,w,h]
        for l in range(8):
            filter_tensor[l, :, :] = self.filters.view(-1)[l]  # construct filtermap uniformly from given filters
        mapped_img = torch.cat((image_tensor.cpu(), filter_tensor.cpu()), dim=0).unsqueeze(dim=0).to(self.device)  # concat filters and img

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        enhanced_img = enhanced_img.cpu()
        enhanced_img = enhanced_img.detach().permute(2, 3, 1, 0).squeeze().numpy()

        enhanced_clipped = np.clip(enhanced_img, 0.0, 1.0) * 255.0
        enhanced_clipped = enhanced_clipped.astype('uint8')
        return enhanced_clipped     # returns a np array

    def re_init(self):
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True, device=self.device)
        if config.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)
        elif config.optim == 'adam':
            self.optimizer = torch.optim.Adam(params=[self.filters], lr=config.optim_lr)
        else:
            error_callback('optimizer')

    #  TODO: autobright automatisch aufrufen???
    def enhance_image(self, image_path, re_init=True, rescale_to_hd=True):  # accepts image_path as string, but also as PIL image object

        if re_init:
            self.re_init()  # not called from NICER button, but from batch mode in optimize_whole_folder -> new filters for each image run
        else:
            user_preset_filters = [self.filters[x].item() for x in range(8)]
        # re-init can be seen as test whether initial filter values (!= 0) should be used or not during optimization

        if isinstance(image_path, str):
            bright_normalized_img = normalize_brightness(image_path)
            pil_image = Image.fromarray(bright_normalized_img)
            #pil_image = load_pil_img(image_path)
        else:
            print("else")
            pil_image = image_path
            bright_normalized_img = normalize_brightness(pil_image, input_is_PIL=True)
            pil_image = Image.fromarray(bright_normalized_img)

        image_tensor_transformed = nima_transform(pil_image)
        image_tensor_transformed_batched = image_tensor_transformed.unsqueeze(dim=0).to(self.device)

        with torch.no_grad():
            initial_nima_score = get_tensor_mean_as_float(self.nima(image_tensor_transformed_batched))
            first_pass_distribution, _ = self.forward(image_tensor_transformed)
            initial_nima_score_after_first_pass = get_tensor_mean_as_float(first_pass_distribution)

        nima_offset = initial_nima_score - initial_nima_score_after_first_pass
        epochs = config.epochs
        losses = []

        filters_for_plot = {}

        # optimize image:
        print_msg("Starting optimization", 2)
        start_time = time.time()
        for i in range(epochs):
            print_msg("Iteration {} of {}".format(i, epochs), 2)
            distribution, enhanced_img = self.forward(image_tensor_transformed)
            self.optimizer.zero_grad()
            if re_init:
                filters_for_plot[i] = [self.filters[x].item() for x in range(8)]
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(), gamma=self.gamma)  # re-init True, i.e. new for each image
                losses.append(loss.item())
            else:
                filters_for_plot[i] = [self.filters[x].item() for x in range(8)]
                loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu(), initial_filters=user_preset_filters, gamma=self.gamma)  # TODO: test, and gamma too
                losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        print_msg("Optimization for %d epochs took %.3fs" % (epochs, time.time() - start_time), 2)

        if config.plot_filter_intensities:
            plot_filter_intensities(filters_for_plot)

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

    def enhance_image_folder(self, folder_path, random=False):
        if not os.path.exists(os.path.join(folder_path, 'results')):
            os.mkdir(os.path.join(folder_path, 'results'))

        no_of_imgs = len([x for x in os.listdir(folder_path) if x.split('.')[-1] in config.supported_extensions])

        results = {}
        for idx, img_name in enumerate(os.listdir(folder_path)):
            img_basename = img_name.split('.')[0]
            extension = img_name.split('.')[-1]
            if extension not in config.supported_extensions: continue
            print_msg("\nWorking on image {} of {}".format(idx, no_of_imgs), 1)

            if random:  # make random destructive baseline
                random_filters = [0.0] * 8
                for i in range(len(random_filters)):
                    random_filters[i] = np.random.uniform(-50, 50) / 100.0  # filter order doesn matter, it's all random anyway
                self.set_filters(random_filters)
                results[img_name + '_init'] = self.filters.tolist()
                init_pil_img = Image.open(os.path.join(folder_path, img_name))
                init_random_img_np = self.single_image_pass_can(init_pil_img, resize=True)
                init_random_img_pil = Image.fromarray(init_random_img_np)
                init_random_img_pil.save(os.path.join(folder_path, 'results', img_basename + '_init.' + extension))
                enhanced_img, init_nima, final_nima = self.enhance_image(os.path.join(folder_path, img_name), re_init=False)

            else:
                enhanced_img, init_nima, final_nima = self.enhance_image(os.path.join(folder_path, img_name))

            pil_img = Image.fromarray(enhanced_img)
            pil_img.save(os.path.join(folder_path, 'results', img_basename + '_enhanced.' + extension))

            results[img_name] = (init_nima, final_nima, self.filters.tolist())

        json.dump(results, open(os.path.join(folder_path, 'results', "results.json"), 'w'))
        print_msg("Saved results. Finished.", 1)




def plot_filter_intensities(intensities_for_plot):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.use('pdf')
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    from scipy.interpolate import make_interp_spline
    x_e = np.arange(1, config.epochs + 1)
    x = np.linspace(1, config.epochs + 1, 500).tolist()
    sat, con, bri, sha, hig, llf, exp, nld = [], [], [], [], [], [], [], []

    for key, val in intensities_for_plot.items():
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

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.14, bottom=0.22, right=0.95, top=0.87)

    h0 = ax.plot(x, a)
    h1 = ax.plot(x, b)
    h2 = ax.plot(x, c)
    h3 = ax.plot(x, d)
    h4 = ax.plot(x, e)
    h5 = ax.plot(x, f)
    h6 = ax.plot(x, g)
    h7 = ax.plot(x, h)

    ax.set_xlabel('Optimization Epochs')
    ax.set_ylabel('Filter Intensity')

    width = 5.487*2
    height = width / 1.218
    fig.set_size_inches(width, height)

    ax.legend((h0[0], h1[0], h2[0], h3[0], h4[0], h7[0], h6[0], h5[0]), ('Sat', 'Con', 'Bri', 'Sha', 'Hig', 'Exp', 'LLF', 'NLD'))

    fig.savefig('results.pdf')
    #plt.show()

    # returns enhanced image as np array
