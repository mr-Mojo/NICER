from imports import *
from utils import *
from neural_models import *

class NICER(nn.Module):

    def __init__(self, checkpoint_can, checkpoint_nima, device, can_arch=8):
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
        self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)

    def forward(self, image):
        filter_tensor = torch.zeros((8, 224, 224), dtype=torch.float32).to(self.device)
        for l in range(8):
            filter_tensor[l, :, :] = self.filters.view(-1)[l]  # construct filtermap uniformly from given filters
        mapped_img = torch.cat((image, filter_tensor), dim=0).unsqueeze(dim=0)  # concat filters and img

        enhanced_img = self.can(mapped_img)  # enhance img with CAN
        distr_of_ratings = self.nima(enhanced_img)  # get nima score distribution -> tensor

        return distr_of_ratings, enhanced_img

    def re_init(self):
        self.filters = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.SGD(params=[self.filters], lr=config.optim_lr, momentum=config.optim_momentum)

    def enhance_image(self, image_path, rescale_to_hd=True):
        self.re_init()

        pil_image = load_pil_img(image_path)
        image_tensor_transformed = nima_transform(pil_image)
        image_tensor_transformed_batched = image_tensor_transformed.unsqueeze(dim=0)

        with torch.no_grad():
            initial_nima_score = get_tensor_mean_as_float(self.nima(image_tensor_transformed_batched))
            first_pass_distribution, _ = self.forward(image_tensor_transformed)
            initial_nima_score_after_first_pass = get_tensor_mean_as_float(first_pass_distribution)

        nima_offset = initial_nima_score - initial_nima_score_after_first_pass
        epochs = config.epochs

        # optimize image:
        print("Starting optimization")
        start_time = time.time()
        for i in range(epochs):
            distribution, enhanced_img = self.forward(image_tensor_transformed)
            self.optimizer.zero_grad()
            loss = loss_with_l2_regularization(distribution.cpu(), self.filters.cpu())
            loss.backward()
            self.optimizer.step()
        print("Optimization for %d epochs took %.3fs" % (epochs, time.time() - start_time))

        if rescale_to_hd:
            pil_image = Image.open(image_path)
            if pil_image.size[0] > config.final_size or pil_image.size[1] > config.final_size:
                original_tensor_transformed = hd_transform(pil_image)
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
