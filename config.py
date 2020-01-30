
# ----- NICER parameters:

filter_count = 8
can_checkpoint_path = 'models/can8_epoch10_final.pt'
nima_checkpoint_path = 'models/nima_vgg_bright2.pkl'

gamma = 0.1
epochs = 2
optim = 'adam'
optim_lr = 0.05
optim_momentum = 0.9

desired_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.09, 0.15, 0.55, 0.20]

# ----- image parameters:

rescale = True
final_size = 1920
supported_extensions = ['jpg', 'jpeg', 'png']       # might work with more, not yet tested
supported_extensions_raw = ['dng']                  # might work with more, not yet tested

# ----- GUI parameters:

plot_filter_intensities = False                     # use only with epochs >= 20
save_filter_intensities = False                     # save intensity plot as pdf
verbosity = 2


# ------ for random enhancement:
save_path_random = 'results'