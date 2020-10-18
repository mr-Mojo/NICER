import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat
from skimage.metrics import structural_similarity as ssim

from utils import print_msg


def get_ssim(img_true, img_manipulated):
    img_true = cv2.resize(img_true, (400, 300))  # resize for speed
    img_manipulated = cv2.resize(img_manipulated, (400, 300))
    return ssim(img_true, img_manipulated, data_range=img_manipulated.max() - img_manipulated.min(), multichannel=True)


def get_psnr(img_true, img_manipulated):
    return cv2.PSNR(img_true, img_manipulated)


def get_brightness(im_path, read=True):
    if read:
        im_file = Image.open(im_path)
    else:
        im_file = im_path  # passed "path" already is PIL image
    stat = ImageStat.Stat(im_file)
    try:
        r, g, b = stat.mean  # RGB
        res = math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
    except ValueError:
        mean = stat.mean  # grayscale
        res = math.sqrt(mean[0])
    return res


def check_if_is_black_image(file, p=60):
    width, height, depth = file.shape
    count = 0
    no_of_samples = int(width * height * 0.05)  # check 5 % of the image pixels
    for i in range(no_of_samples):
        test = file[np.random.randint(0, width)][np.random.randint(0, height)]
        if test[0] < 10 and test[1] < 10 and test[2] < 10:
            count += 1
    res = (count / no_of_samples) * 100.0
    if res > p:
        return True
    else:
        return False


def check_if_is_white_image(file, p=60):
    width, height, depth = file.shape
    count = 0
    no_of_samples = int(width * height * 0.05)  # check 5 % of the image pixels
    for i in range(no_of_samples):
        test = file[np.random.randint(0, width)][np.random.randint(0, height)]
        if test[0] > 250 and test[1] > 250 and test[2] > 250:
            count += 1
    res = (count / no_of_samples) * 100.0
    if res > p:
        return True
    else:
        return False


def rescale_hsv(img, value):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (255 - hsv_img[:, :, 2]) < value
    hsv_img[:, :, 2] = np.where(mask, 255, hsv_img[:, :, 2] + value)
    hsv_img[:, :, 2] = np.where(hsv_img[:, :, 2] < 0, 0, hsv_img[:, :, 2])
    bright_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return bright_img


def auto_bright(image, clip_hist_percent=5.0, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    isRaw = False

    # Calculate grayscale histogram
    if np.amax(gray) > 2.0:  # int format, no raw
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
    else:
        isRaw = True  # float format, raw
        hist = cv2.calcHist([gray * 256], [0], None, [65536], [0, 65536])
        hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    if isRaw:
        alpha = 65536 / (maximum_gray - minimum_gray)  # contrast adjustment
    else:
        alpha = 255 / (maximum_gray - minimum_gray)  # contrast adjustment

    beta = -minimum_gray * alpha  # brightness adjustment

    # Calculate new histogram with desired range and show histogram
    if plot:
        new_hist = cv2.calcHist([gray], [0], None, [256], [minimum_gray, maximum_gray])
        k1 = plt.plot(hist)
        k2 = plt.plot(new_hist)
        plt.legend((k1[0], k2[0]), ('old', 'new'))
        plt.xlim([0, 256])
        plt.show()

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


# returns a cv2.image that has been brightness normalized
def normalize_brightness(img_path, input_is_PIL=False, verbose=False):
    if input_is_PIL == False:
        img = cv2.imread(img_path)
        brightness = get_brightness(img_path)
    else:
        img = np.array(img_path)  # convert PIL to np to opencv
        brightness = get_brightness(img_path, read=False)  # brightness uses PIL
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bright_img = None

    print_msg("ABN - PIL brightness: {}".format(brightness), 2)
    # best brightness is 128, allow margin of +-30

    # image too dark
    if brightness < 90 or (95 > brightness > 85):

        # image too dark, with lots of blacks: use ScaleAbs, bc HSV shift produces artifacts
        if brightness < 33:
            bright_img = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
            print_msg("ABN - converted a very dark image with ScaleAbs", 2)

        # image too dark, but not as much black: use HSV shift
        elif brightness < 70:

            if check_if_is_black_image(img):
                bright_img = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
                print_msg("ABN - converted a dark image with ScaleAbs", 2)
            else:

                shift = 20
                bright_img = rescale_hsv(img, value=shift)
                psnr = get_psnr(img, bright_img)

                while psnr < 30.0:  # 30 bc then images look "nice" enough -- hard coded
                    print_msg("ABN - hsv stretch psnr is {} with shift of {}".format(psnr, shift), 3)
                    shift -= 2
                    bright_img = rescale_hsv(img, value=shift)
                    psnr = get_psnr(img, bright_img)
                    if shift < 1.0: break

                print_msg("ABN - corrected a dark image with HSV shift", 2)

        # brightness between 70 and 95 -> just a little dark, apply slight scaleAbs
        else:
            num = 1.3
            bright_img = cv2.convertScaleAbs(img, alpha=num, beta=0)
            print_msg("ABN - corrected a (little too) dark image with ScaleAbs", 2)

    # image too bright
    elif brightness > 150:

        # image is too bright, but does not have many whites
        if not check_if_is_white_image(img):
            print_msg("ABN - Auto stretch a bright image", 2)
            clipping = 5.0
            bright_img, alpha, beta = auto_bright(img, plot=False, clip_hist_percent=clipping)

            bright_img_resized = cv2.resize(bright_img, (400, 300))  # resize for ssim speed
            orig_img_resized = cv2.resize(img, (400, 300))

            ssim = get_ssim(orig_img_resized, bright_img_resized)
            if ssim < 0.80:  # not good enough, clip harder
                clipping = 1.0
                while ssim < 0.80:
                    print_msg("ABN - auto-bright now with clip of {} bc ssim is {}".format(clipping, ssim), 3)
                    bright_img_resized, _, _ = auto_bright(orig_img_resized, clip_hist_percent=clipping)
                    ssim = get_ssim(orig_img_resized, bright_img_resized)
                    clipping /= 10.0
                    if clipping < 1e-6:
                        break

            # apply found clipping value on full size image
            bright_img, _, _ = auto_bright(img, plot=False, clip_hist_percent=clipping)

        # image too bright, but contains lots of white -> probably should be bright, do nothing
        else:
            print_msg("ABN - do nothing - bright white image, kept it bright", 2)

    # image brightness in range 95 < brightness < 150: brightness is okay!
    else:
        print_msg("ABN - no ABN enhancement - image brightness is good.", 2)

    if bright_img is None:
        if input_is_PIL == False:
            bright_img = cv2.imread(img_path)
        else:
            cv2_img = np.array(img_path)  # convert PIL to np to opencv
            bright_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

    # else:
    # bright_img_np = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
    # bright_img_final = Image.fromarray(bright_img_np)

    # returns the image in RGB
    bright_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
    return bright_img


def correct_image_folder(path, save_corrected=True, verbose=False, resize=False, resize_factor=0.5,
                         show_output=False, extension='.jpg'):
    """
    Will correct all images in a given folder that end with extension and need correction,
    and save them to a new folder named /corrected within the same directory

    @ params:
    -----------------------------------------
    path: string, path of the image folder that should be corrected
    save_corrected: bool, whether to save the corrected images
    verbose: bool, whether to print messages during correction
    resize: bool, whether to resize the images for processing and saving. Recommended for large files.
    resize_factor: float, factor with which the original image dimensions will get multiplied when resizing
    show_output: bool, show the corrected output for each image
    extension: string, all images that have this extension will be processed

    @ returns:
    -----------------------------------------
    None, but saves all images into /corrected when save_output=True
    """

    for idx, img_name in enumerate(sorted(os.listdir(path))):
        if not img_name.endswith(extension): continue

        print("Normalizing img {} of {}".format(idx, len(os.listdir(path))))

        # currently unavailable
        # if resize:
        #     try:
        #         width, height, depth = img.shape  # RGB
        #     except ValueError:
        #         width, height = img.shape  # greyscale
        #     new_width = int(width * resize_factor)
        #     new_height = int(height * resize_factor)
        #     img = cv2.resize(img, (new_height, new_width))

        bright_img = normalize_brightness(os.path.join(path, img_name))

        # brightness correction finished.
        # display and print results of correction:

        if bright_img is not None and verbose:
            print("PIL brightness (new): \t\t", get_brightness(Image.fromarray(bright_img), read=False))

        if bright_img is not None and show_output:
            cv2.imshow("brightened", bright_img)
        elif show_output:
            img = cv2.imread(os.path.join(path, img_name))
            cv2.imshow("original", img)
            cv2.waitKey()

        if bright_img is not None and save_corrected:
            dest = os.path.join(path, 'corrected')
            if not os.path.exists(dest):
                os.mkdir(dest)
            bright_img = cv2.cvtColor(bright_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dest, img_name), bright_img)

    print("Done.")
