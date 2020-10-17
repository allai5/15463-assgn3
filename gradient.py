"""
Implementations of:

References:
    - http://hhoppe.com/flash.pdf
"""
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter
from scipy import signal
import cv2
import matplotlib.pyplot as plt

class GradientProcess():
    def __init__(self):
        pass

    def gradient_channel(self, imgc):
        kernel_x = [[1, -1]]
        kernel_y = [[1],[-1]]

        img_dx = signal.convolve2d(imgc, kernel_x, mode='same')
        img_dxx = signal.convolve2d(img_dx, kernel_x, mode='same')

        img_dy = signal.convolve2d(imgc, kernel_y, mode='same')
        img_dyy = signal.convolve2d(img_dx, kernel_y, mode='same')

        return np.add(img_dxx, img_dyy)

    def Laplacian_filter(self, imgc):
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        return signal.convolve2d(imgc, lap_kernel, mode='same')

    def set_boundary(self, Ic, imgc):
        Ic[0:,]  = imgc[0:,]
        Ic[-1:,] = imgc[-1:,]
        Ic[:,0]  = imgc[:,0]
        Ic[:,-1] = imgc[:,-1]

        return Ic

    def poisson_solve(self, divI, Is_c_init, eps, N, imgc):
        print("POISSON SOLVE")
        # Initialization
        Is_c = self.set_boundary(Is_c_init, imgc)
        r = divI - self.Laplacian_filter(Is_c)
        d = r
        delta_new = np.inner(r, r)
        n = 0

        # Conjugate gradient descent iteration
        while (np.linalg.norm(r) > eps and n < N):
            q = self.Laplacian_filter(d)
            eta = delta_new / np.inner(d, q)
            Is_c = self.set_boundary(Is_c + np.matmul(eta, d), imgc)


            r = r - np.matmul(eta, q)
            delta_old = delta_new
            delta_new = np.inner(r, r)
            beta = delta_new / delta_old

            nan_ids = np.argwhere(np.isnan(beta))
            for nid in nan_ids:
                beta[nid] = 0

            d = r + np.matmul(beta, d)
            n += 1
            print(n)

        return Is_c

    def diff_and_integrate(self):
        img = io.imread("data/museum/museum_ambient.png") / 255.0

        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        eps = 0.1
        N = 2000

        divI_r = self.gradient_channel(img_r)
        divI_g = self.gradient_channel(img_g)
        divI_b = self.gradient_channel(img_b)

        Is_r_init = np.zeros((img_r.shape[0], img_r.shape[1]))
        Is_g_init = np.zeros((img_r.shape[0], img_r.shape[1]))
        Is_b_init = np.zeros((img_r.shape[0], img_r.shape[1]))

        Is_r = self.poisson_solve(divI_r, Is_r_init, eps, N, img_r)
        Is_g = self.poisson_solve(divI_g, Is_g_init, eps, N, img_g)
        Is_b = self.poisson_solve(divI_b, Is_b_init, eps, N, img_b)

        Is = np.dstack((Is_r, Is_g, Is_b))
        print(Is)
        fig = plt.figure()
        fig.add_subplot(2,1,1)
        plt.imshow(img)
        fig.add_subplot(2,1,2)
        plt.imshow(Is)
        plt.show()

        return Is

