import os
import csv
import sys
import time
import math
import json
import torch
import rawpy
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image, ImageStat
import torchvision.models as models
from collections.abc import Iterable
from skimage.transform import resize
from torchvision.transforms import transforms
from skimage.metrics import structural_similarity as ssim


try:
    import cv2
except ImportError:
    print("Cannot import OpenCV.")
    # TODO: eventuell import irgendwas as cv2, einfach damit's kompiliert falls es damit Probleme gibt.