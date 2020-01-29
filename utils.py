from imports import *
import config

def error_callback(caller):
    if caller in ['mae', 'mse', 'mae_channelwise', 'ssim', 'psnr']:
        sys.exit("Exit - " + caller + " - shapes do not match")
    elif caller is 'filter_index' or caller is 'filter_value':
        sys.exit("given " + caller + " cannot be resolved")
    elif caller is 'forward_conv':
        sys.exit("Convolution does not preserve resolution - shape mismatch in model forward")
    elif caller is 'raw_img':
        sys.exit("Can only output 8 or 16 bit images")
    elif caller is 'emd_loss':
        sys.exit("Distribution shapes do not match in EMD loss")
    elif caller is 'filter_length_l2loss':
        sys.exit("Filter lengths do not match.")
    elif caller is 'optimizer':
        sys.exit("Illegal optimizer. Use SGD or ADAM.")


nima_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


hd_transform = transforms.Compose([     # use before saving the big image, to avoid out of memory errors
    transforms.Resize(config.final_size),       # smaller edge will be matched to this
    transforms.ToTensor()
])


def load_pil_img(path):
    img = Image.open(path)
    return img


def get_tensor_mean_as_tensor(nima_distribution):     # returns a tensor!
    out = nima_distribution.view(10, 1)
    mean = 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
    return mean


def get_tensor_mean_as_float(nima_distribution):     # returns a float!
    tensor_result = get_tensor_mean_as_tensor((nima_distribution))
    return tensor_result.item()


def print_msg(message, level):
    if level <= config.verbosity:
        print(message)

def get_filter_index(filter_name):
    if filter_name == 'sat':
        return 0
    elif filter_name == 'con':
        return 1
    elif filter_name == 'bri':
        return 2
    elif filter_name == 'sha':
        return 3
    elif filter_name == 'hig':
        return 4
    elif filter_name == 'llf':
        return 5
    elif filter_name == 'nld':
        return 6
    elif filter_name == 'exp':
        return 7
    else:
        error_callback('filter_index')


def read_raw_img(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(output_bps=16)
    rgb_img = rgb.astype(np.float32) / 65536.0
    return rgb_img


def get_tensor_from_raw_image(path, size=None):
    rgb_float = read_raw_img(path)
    if size:
        if isinstance(size, Iterable):
            rgb_float_resized = resize(rgb_float, (224,224))
        else:
            # size was given as 1 number: match longer side if it exceeds size, else leave it small as it is
            width, height, depth = rgb_float.shape
            if width > size or height > size:
                if width > height:
                    factor = size / width     # width * factor = 1080 --> factor = 1080/width
                else:
                    factor = size / height
                new_width = int(width * factor)
                new_height = int(height * factor)
                rgb_float_resized = resize(rgb_float, (new_width, new_height)) # resize: (rows, cols)
            else:
                rgb_float_resized = rgb_float
    else:
        rgb_float_resized = rgb_float

    img_tensor = transforms.ToTensor()(rgb_float_resized)
    return img_tensor


def single_emd_loss(p, q, r=2):
    if not p.shape == q.shape: error_callback('emd_loss')
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def loss_with_l2_regularization(nima_result, filters, gamma=config.gamma, initial_filters=None):
    if initial_filters is not None:
        if len(filters) != len(initial_filters): error_callback('filter_length_l2loss')

    desired_distribution = torch.FloatTensor(config.desired_distribution).view((-1, 10))
    distance_term = sum(single_emd_loss(desired_distribution, nima_result))

    if initial_filters is not None:
        filter_deviations_from_initial = sum([(filters[x].item() - initial_filters[x]) ** 2 for x in range(len(filters))])  # l2: sum the deviation from user preset
        l2_term = filter_deviations_from_initial
        print_msg("\nInitial Filters: {}".format(initial_filters),3)
        print_msg("Current Filters: {}".format([filters[x].item() for x in range(8)]), 3)
        print_msg("Deviation from Initial: {}".format(filter_deviations_from_initial), 3)
        print_msg("L2 Term: {}".format(l2_term), 3)
    else:
        l2_term = sum([fil**2 for fil in filters])                      # l2: sum the squares of all filters

    return distance_term + gamma*l2_term


def read_raw_img_into_tensor(path, size=None):
    pass
