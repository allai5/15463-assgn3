"""
Main wrapper function for bilateral filter denoising and gradient-domain
processing fusion algorithm

"""
from filter import BilateralFilter
from gradient import GradientProcess
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
# apath = "data/lamp/lamp_ambient.tif"
# fpath = "data/lamp/lamp_flash.tif"
apath = "data/museum/museum_ambient.png"
fpath = "data/museum/museum_flash.png"
# apath = "Task3_photos/cow_bowl_ambient.JPG"
# fpath = "Task3_photos/cow_bowl_flash.JPG"

def main():
    bf = BilateralFilter(apath, fpath)

    gp = GradientProcess(apath, fpath)
    gp.fuse_gradient_test()

if __name__ == "__main__":
    main()
