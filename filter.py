"""
Implementations of Piecewise Bilateral Filter and Joint Bilateral Filter

Reference: http://hhoppe.com/flash.pdf
"""
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt

class BilateralFilter():
    def __init__(self, apath, fpath, sigs, sigr):
        self.apath = apath
        self.fpath = fpath
        self.imga = (io.imread(apath)) / 255.0
        self.flat_dim = self.imga.shape[0] * self.imga.shape[1]

        self.img_rc = self.imga[:,:,0]
        self.img_gc = self.imga[:,:,1]
        self.img_bc = self.imga[:,:,2]

        # range 0 ... 255
        self.rmin = np.min(self.img_rc)
        self.gmin = np.min(self.img_gc)
        self.bmin = np.min(self.img_bc)

        self.rmax = np.max(self.img_rc)
        self.gmax = np.max(self.img_gc)
        self.bmax = np.max(self.img_bc)

        self.sigs  = 25.0
        # self.sigs  = 3.0
        self.sigr  = 0.1

        self.r_segs = int((self.rmax - self.rmin) / self.sigr)
        self.g_segs = int((self.gmax - self.gmin) / self.sigr)
        self.b_segs = int((self.bmax - self.bmin) / self.sigr)
        self.prev_weights = np.zeros((self.imga.shape[0], self.imga.shape[1]))

    def gr(self, mu, sigma):
        return np.random.normal(mu, sigma)

    def get_weights(self, imgc, ij):
        ij0 = ij - self.sigr
        ij2 = ij + self.sigr
        wj  = (ij2  - imgc) / self.sigr
        wj0 = (imgc - ij0) / self.sigr

        weights_j   = np.where((ij <= imgc) & (imgc < ij2), wj, 0)
        weights_j0  = np.where((ij0 <= imgc) & (imgc < ij), wj0, 0)

        weights = np.add(weights_j, weights_j0)
        # print(np.add(self.prev_weights, weights_j0))
        self.prev_weights = weights_j
        # print(weights)

        return weights

    def filter_channel(self, channel):
        print("filter channel")
        imgc = self.img_rc
        segs = self.r_segs
        cmax = self.rmax
        cmin = self.rmin

        if(channel == 1):
            imgc = self.img_gc
            segs = self.g_segs
            cmax = self.gmax
            cmin = self.gmin

        if(channel == 2):
            imgc = self.img_bc
            segs = self.b_segs
            cmax = self.bmax
            cmin = self.bmin

        J = np.zeros((imgc.shape[0], imgc.shape[1]))
        dim = int(3.0 * self.sigs)
        for j in range(segs + 1):
            print(str(j) + "/" + str(segs))
            ij = cmin + j * (cmax - cmin)/segs
            ij1 = ij + self.sigr
            Gj = self.gr(imgc - ij, self.sigr)

            Kj = cv2.GaussianBlur(Gj, (dim, dim), self.sigs) / 255.0
            Hj = np.multiply(Gj, imgc)
            Hsj = cv2.GaussianBlur(Hj, (dim, dim), self.sigs) / 255.0
            # print(Hsj)
            # print(np.count_nonzero(np.subtract(Hsj, Hj)))
            Jj = np.divide(Hsj, Kj)

            min_value = np.nanmin(Jj)
            nan_ids = np.argwhere(np.isnan(Jj))
            for nid in nan_ids:
                Jj[nid] = 0

            J = np.add(J, np.multiply(Jj, self.get_weights(imgc, ij)))
            # print(np.max(J))

        return J

    def filter(self):
        frc = self.filter_channel(0)
        fgc = self.filter_channel(1)
        fbc = self.filter_channel(2)

        print(np.max(frc), np.max(fgc), np.max(fbc))
        img_bf = np.dstack((frc, fgc, fbc))
        plt.imshow(np.clip(img_bf, 0, 1), cmap='gray')
        plt.show()
        return img_bf

# class JointBilateralFilter():

