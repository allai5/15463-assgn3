from filter import BilateralFilter
from gradient import GradientProcess
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

    gp.diff_and_integrate()


if __name__ == "__main__":
    main()
