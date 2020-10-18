# ----- application parameters:

verbosity = 2                                      # set from 0 (silent) to 3 (most verbose)
preview = True                                     # toggle to true to show preview imgs -> slower

gamma = 0.1
epochs = 50
optim = 'sgd'                                      # also supports adam
optim_lr = 0.05
optim_momentum = 0.9

# ----- image parameters:

rescale = True
final_size = 1920                                   # final size when saving. Attention: bigger imgs require more memory
supported_extensions = ['jpg', 'jpeg', 'png']
supported_extensions_raw = ['dng']                  # legacy, deprecated

# ----- Architecture parameters:
# (you should not have to change these)

can_filter_count = 8
can_checkpoint_path = 'models/can8_epoch10_final.pt'
nima_checkpoint_path = 'models/nima_vgg_bright2.pkl'

desired_distribution = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.09, 0.15, 0.55, 0.20]

