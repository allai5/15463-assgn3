"""
Implementations of:
    - Piecewise Bilateral Filter
    - Joint Bilateral Filter
    - Detail Transfer
    - Shadow/Specularity Mask

References:
    - http://hhoppe.com/flash.pdf
    - https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf
"""
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt

class BilateralFilter():
    def __init__(self, apath, fpath):
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
            # Gj = self.gr(imgc - ij, sigr)
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
        return img_filter

    def joint_filter(self, sigs, sigr):
        joint_r = self.filter_channel(0, 1, 0, sigs, sigr)
        joint_g = self.filter_channel(1, 1, 0, sigs, sigr)
        joint_b = self.filter_channel(2, 1, 0, sigs, sigr)

        img_joint = np.dstack((joint_r, joint_g, joint_b))
        return img_joint

    def detail_transfer(self, eps):
        Fbase = np.clip(io.imread("flash_bilateral_filter.png"), 0, 255) / 255.0
        Anr = np.clip(io.imread("joint_filter.png"), 0, 255) / 255.0

        detail = np.divide((self.imgf + eps), (Fbase + eps))
        nan_ids = np.argwhere(np.isnan(detail))
        min_value = np.min(detail)
        detail = np.nan_to_num(detail)
        detail = np.clip(detail, 0, 1)

        Adet = np.multiply(Anr, detail)
        Adet = np.clip(Adet, 0, 1)

        return Adet

    def linearize_img(self, C_nonlin):

        C_lin = np.where((C_nonlin <= 0.0404482), C_nonlin / 12.92,
                         np.power((C_nonlin + 0.055)/1.055, 2.4))
        return C_lin

    def shadow_mask(self, thresh):
        Alin = self.linearize_img(io.imread("data/lamp/lamp_ambient.tif")/255.0)
        Flin = self.linearize_img(io.imread("data/lamp/lamp_flash.tif")/255.0)

        Ar = Alin[:,:,0]
        Ag = Alin[:,:,1]
        Ab = Alin[:,:,2]

        Fr = Flin[:,:,0]
        Fg = Flin[:,:,1]
        Fb = Flin[:,:,2]

        r_mask = np.where(Fr - Ar <= thresh, 1.0, 0.0)
        g_mask = np.where(Fg - Ag <= thresh, 1.0, 0.0)
        b_mask = np.where(Fb - Ab <= thresh, 1.0, 0.0)

        return r_mask, g_mask, b_mask

    def specular_mask(self):
        Alin = self.linearize_img(io.imread("data/lamp/lamp_ambient.tif")/255.0)
        Flin = self.linearize_img(io.imread("data/lamp/lamp_flash.tif")/255.0)

        Ar = Alin[:,:,0]
        Ag = Alin[:,:,1]
        Ab = Alin[:,:,2]

        Fr = Flin[:,:,0]
        Fg = Flin[:,:,1]
        Fb = Flin[:,:,2]

        r_mask = np.where(Fr <= 0.95, 0.0, 1.0)
        g_mask = np.where(Fg <= 0.95, 0.0, 1.0)
        b_mask = np.where(Fb <= 0.95, 0.0, 1.0)

        return r_mask, g_mask, b_mask


    def apply_mask(self, shadow_thresh):
        Adet = io.imread("detail_transfer.png") / 255.0
        Abase = io.imread("bilateral_filter.png") / 255.0
        r_shadowmask, g_shadowmask, b_shadowmask = self.shadow_mask(shadow_thresh)
        r_specularmask, g_specularmask, b_specularmask = self.specular_mask()

        r_mask = np.logical_or(r_shadowmask, r_specularmask).astype(float)
        g_mask = np.logical_or(g_shadowmask, g_specularmask).astype(float)
        b_mask = np.logical_or(b_shadowmask, b_specularmask).astype(float)

        mask = np.dstack((r_mask, g_mask, b_mask))
        plt.imshow(mask)
        plt.show()
        Afinal = np.add(np.multiply((1.0 - mask), Adet), np.multiply(mask, Abase))
        return Afinal
