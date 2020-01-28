
# ----- NICER parameters:

filter_count = 8
can_checkpoint_path = 'models/can_exp9_final.pt'
nima_checkpoint_path = 'models/nima_vgg_bright2.pkl'

gamma = 0.1
epochs = 50
optim = 'sgd'
optim_lr = 0.05
optim_momentum = 0.9

desired_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.09, 0.15, 0.55, 0.20]

# ----- image parameters:

final_size = 1920
supported_extensions = ['jpg', 'png']
supported_extensions_raw = ['dng']

# ----- GUI parameters:

plot_filter_intensities = False
verbosity = 2


# ------ for random enhancement:
save_path_random = 'results'