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
        self.imgf = (io.imread(fpath)) / 255.0

        self.prev_weights = np.zeros((self.imga.shape[0], self.imga.shape[1]))

    def gr(self, mu, sigma):
        return np.exp(-np.square(mu) / (2.0 * np.square(sigma)))

    def get_weights(self, imgc, ij, sigr):
        ij0 = ij - sigr
        ij2 = ij + sigr
        wj  = (ij2  - imgc) / sigr
        wj0 = (imgc - ij0) / sigr

        weights_j   = np.where((ij <= imgc) & (imgc < ij2), wj, 0)
        weights_j0  = np.where((ij0 <= imgc) & (imgc < ij), wj0, 0)

        weights = np.add(weights_j, weights_j0)
        # print(np.add(self.prev_weights, weights_j0))
        self.prev_weights = weights_j
        # print(weights)

        return weights

    def filter_channel(self, channel, joint, flash, sigs, sigr):
        print("filter channel")
        if (flash): imgc = self.imgf[:,:,channel]
        else: imgc = self.imga[:,:,channel]

        imgcf = self.imgf[:,:,channel]
        cmax = np.max(imgc)
        cmin = np.min(imgc)
        segs = int((cmax - cmin) / sigr)

        J = np.zeros((imgc.shape[0], imgc.shape[1]))
        kernel_dim = int(3 * sigs)

        for j in range(segs + 1):
            ij = cmin + j * (cmax - cmin)/segs
            Gj = self.gr(imgc - ij, sigr)
            if (joint):
                Gj = self.gr(imgcf - ij, sigr)
            else:
                Gj = self.gr(imgc - ij, sigr)

            Kj = cv2.GaussianBlur(Gj, (kernel_dim, kernel_dim), sigs)
            Hj = np.multiply(Gj, imgc)
            Hsj = cv2.GaussianBlur(Hj, (kernel_dim, kernel_dim), sigs)
            Jj = np.divide(Hsj, Kj)

            nan_ids = np.argwhere(np.isnan(Jj))
            for nid in nan_ids:
                Jj[nid] = 0

            if (joint):
                weights = self.get_weights(imgcf, ij, sigr)
            else:
                weights = self.get_weights(imgc, ij, sigr)

            J = np.add(J, np.multiply(Jj, weights))

        return J

    def bilateral_filter(self, flash, sigs, sigr):
        filter_r = self.filter_channel(0, 0, flash, sigs, sigr)
        filter_g = self.filter_channel(1, 0, flash, sigs, sigr)
        filter_b = self.filter_channel(2, 0, flash, sigs, sigr)

        img_filter = np.dstack((filter_r, filter_g, filter_b))

        plt.imshow(img_filter)
        plt.show()

        return img_filter

    def joint_filter(self, sigs, sigr):
        joint_r = self.filter_channel(0, 1, 0, sigs, sigr)
        joint_g = self.filter_channel(1, 1, 0, sigs, sigr)
        joint_b = self.filter_channel(2, 1, 0, sigs, sigr)

        img_joint = np.dstack((joint_r, joint_g, joint_b))

        plt.imshow(img_joint)
        plt.show()

        return img_joint

    def detail_transfer(self):
        eps = 0.0001
        # Fbase = self.bilateral_filter(1, 25, 0.05)
        # Anr = self.joint_filter(25, 0.05)
        Fbase = np.clip(io.imread("flash_bilateral_filter.png"), 0, 255) / 255.0
        Anr = np.clip(io.imread("joint_filter.png"), 0, 255) / 255.0

        detail = np.divide((self.imgf + eps), (Fbase + eps))
        plt.imshow(self.imgf + eps)
        plt.show()

        nan_ids = np.argwhere(np.isnan(detail))
        print(detail)
        min_value = np.min(detail)
        for nid in nan_ids:
            detail[nid] = min_value;

        detail = np.clip(detail, 0, 1)

        plt.imshow(detail)
        plt.show()
        Adet = np.multiply(Anr, detail)
        Adet = np.clip(Adet, 0, 1)


        # fig = plt.figure()
        # fig.add_subplot(2,1,1)
        # plt.imshow(Anr)
        # fig.add_subplot(2,1,2)
        plt.imshow(Adet)
        plt.show()
        io.imsave("detail_transfer.png", Adet)
        return Adet

    def get_mask(self, thresh):
        Alin_p = io.imread("")
        Flin = io.imread("")
        Alin = Alin_p * (() / ())

        mask = np.where(Flin - Alin <= thresh, 1, 0)
        return mask

    def apply_mask(self):
        Adet = self.detail_transfer()
        mask = self.get_mask(0.05)
        Afinal = (1.0 - mask)*Adet + mask*self.imga

# class JointBilateralFilter():

