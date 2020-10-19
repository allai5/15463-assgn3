from filter import BilateralFilter
from gradient import GradientProcess
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
apath = "data/lamp/lamp_ambient.tif"
fpath = "data/lamp/lamp_flash.tif"
sigs  = 10.0
sigr  = 10.0

def main():
    # bf = BilateralFilter(apath, fpath, sigs, sigr)
    # A = bf.bilateral_filter(0, 25, 0.05)
    # io.imsave("bilateral_filter.png", A)
    # Fbase = bf.bilateral_filter(1, 25, 0.05)
    # io.imsave("flash_bilateral_filter.png", Fbase)
    # Anr = bf.joint_filter(3, 0.05)
    # io.imsave("joint_filter.png", Anr)
    # bf.detail_transfer()
    # bf.apply_mask()
    # bf.shadow_mask(0.1)


    gp = GradientProcess()
    # I = gp.gradient_field_test()
    # It = io.imread("data/museum/museum_ambient.png") / 255.0
    # ItRGB = np.dstack((It[:,:,0], It[:,:,1],  It[:,:,2]))
    # diff = np.subtract(I, ItRGB)
    # print(diff)
    # plt.imshow(diff)
    # plt.show()

    gp.fuse_gradient_test()
    # myimg = io.imread("final_museum.png")/255.0
    # theirimg = io.imread("data/museum/museum_flash.png")/255.0
    # img3d = np.dstack((theirimg[:,:,0], theirimg[:,:,1], theirimg[:,:,2]))
    # diff = np.subtract(myimg, img3d)
    # print(diff)
    # plt.imshow(diff)
    # plt.show()



    # img = io.imread("data/museum/museum_flash.png")/255.0
    # gp.gradient_field(img)

if __name__ == "__main__":
    main()
