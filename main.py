from filter import BilateralFilter
from skimage import io
apath = "data/lamp/lamp_ambient.tiff"
fpath = "data/lamp/lamp_flash.tiff"
sigs  = 10.0
sigr  = 10.0

def main():
    bf = BilateralFilter(apath, fpath, sigs, sigr)
    # Fbase = bf.bilateral_filter(1, 25, 0.05)
    # io.imsave("flash_bilateral_filter.png", Fbase)
    # Anr = bf.joint_filter(5, 0.05)
    # io.imsave("joint_filter.png", Anr)
    # bf.detail_transfer()
    bf.apply_mask()


if __name__ == "__main__":
    main()
