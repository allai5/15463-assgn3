from filter import BilateralFilter

apath = "data/lamp/lamp_ambient.tif"
fpath = "data/lamp/lamp_flash.tif"
sigs  = 10.0
sigr  = 10.0

def main():
    bf = BilateralFilter(apath, fpath, sigs, sigr)
    bf.filter()


if __name__ == "__main__":
    main()
