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
        # lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
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
        # print(divI)
        # print(self.Laplacian_filter(Is_c))
        r = divI - self.Laplacian_filter(Is_c)
        # print(r)
        d = r
        delta_new = np.sum(r * r)
        n = 0

        # Conjugate gradient descent iteration
        while (np.linalg.norm(r) > eps and n < N):
            print(n)
            q = self.Laplacian_filter(d)
            eta = np.divide(delta_new, np.sum(d * q))
            # print(np.matmul(eta,d))
            Is_c = self.set_boundary((Is_c + eta * d), imgc)

            r = r - eta * q
            delta_old = delta_new
            delta_new = np.sum(r * r)
            beta = np.divide(delta_new, delta_old)

            d = r + beta * d
            n += 1

        return Is_c

    def gradient_field_test(self):
        img = io.imread("data/museum/museum_ambient.png") / 255.0

        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        img_rgb = np.dstack((img_r, img_g, img_b))

        eps = 0.01
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

        return Is

    def div_x(self, imgc):
        kernel_x = [[1, -1]]
        img_dx = signal.convolve2d(imgc, kernel_x, mode='same')
        return img_dx

    def div_y(self, imgc):
        kernel_y = [[1],[-1]]
        img_dy = signal.convolve2d(imgc, kernel_y, mode='same')
        return img_dy


    def goc_map(self, imgc, f_imgc):
        fx = self.div_x(f_imgc)
        ax = self.div_x(imgc)

        fy = self.div_y(f_imgc)
        ay = self.div_y(imgc)

        map_top = np.abs(np.add(np.multiply(fx, ax), np.multiply(fy, ay)))
        map_top = np.abs(fx * ax + fy * ay)
        map_bottom = np.sqrt(np.square(fx) + np.square(fy)) * \
                     np.sqrt(np.square(ax) + np.square(ay))

        map = map_top / map_bottom
        map = np.nan_to_num(map)
        return map

    # sigma = 40, tau_s = 0.9
    def saturation_map(self, f_imgc, sigma=40, tau_s=0.9):
        w = np.tanh(sigma * (f_imgc - tau_s))
        norm = np.linalg.norm(w)
        w_norm = w / norm

        return w

    def gradient_field(self, img):
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        div_rx = self.div_x(img_r)
        div_ry = self.div_y(img_r)
        div_gx = self.div_x(img_g)
        div_gy = self.div_y(img_g)
        div_bx = self.div_x(img_b)
        div_by = self.div_y(img_b)

        div_x = np.dstack((div_rx, div_gx, div_bx))
        div_y = np.dstack((div_ry, div_gy, div_by))

        io.imsave("fdiv_x.png", div_x)
        io.imsave("fdiv_y.png", div_y)

    def fuse_gradient_channel(self, ws_c, ax_c, ay_c, M_c, fx_c, fy_c):

        gradFsx_term1 = np.multiply(ws_c, ax_c)
        gradFsx_term2a = np.multiply((1.0 - ws_c), (ax_c))
        gradFsx_term2b = np.multiply(M_c, fx_c) + np.multiply((1.0 - M_c), ax_c)

        gradFsx = gradFsx_term1 + np.multiply(gradFsx_term2a, gradFsx_term2b)
        print(gradFsx)

        gradFsy_term1 = np.multiply(ws_c, ay_c)
        gradFsy_term2a = np.multiply((1.0 - ws_c), (ay_c))
        gradFsy_term2b = np.multiply(M_c, fy_c) + np.multiply((1.0 - M_c), ay_c)

        gradFsy = gradFsy_term1 + np.multiply(gradFsy_term2a, gradFsy_term2b)

        return np.dstack((gradFsx, gradFsy))


    def fuse_gradient_test(self):
        img = io.imread("data/museum/museum_ambient.png") / 255.0
        f_img = io.imread("data/museum/museum_flash.png") / 255.0

        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]

        f_img_r = f_img[:,:,0]
        f_img_g = f_img[:,:,1]
        f_img_b = f_img[:,:,2]

        ax_r = self.div_x(img_r)
        ay_r = self.div_y(img_r)
        fx_r = self.div_x(f_img_r)
        fy_r = self.div_y(f_img_r)

        ax_g = self.div_x(img_g)
        ay_g = self.div_y(img_g)
        fx_g = self.div_x(f_img_g)
        fy_g = self.div_y(f_img_g)

        ax_b = self.div_x(img_b)
        ay_b = self.div_y(img_b)
        fx_b = self.div_x(f_img_b)
        fy_b = self.div_y(f_img_b)

        Mr = self.goc_map(img_r, f_img_r)
        Mg = self.goc_map(img_g, f_img_g)
        Mb = self.goc_map(img_b, f_img_b)

        ws_r = self.saturation_map(f_img_r)
        ws_g = self.saturation_map(f_img_g)
        ws_b = self.saturation_map(f_img_b)

        gradFs_r = self.fuse_gradient_channel(ws_r, ax_r, ay_r, Mr, fx_r, fy_r)
        gradFs_g = self.fuse_gradient_channel(ws_g, ax_g, ay_g, Mg, fx_g, fy_g)
        gradFs_b = self.fuse_gradient_channel(ws_b, ax_b, ay_b, Mb, fx_b, fy_b)

        divFs_r = self.div_x(gradFs_r[:,:,0]) + self.div_y(gradFs_r[:,:,1])
        divFs_g = self.div_x(gradFs_g[:,:,0]) + self.div_y(gradFs_g[:,:,1])
        divFs_b = self.div_x(gradFs_b[:,:,0]) + self.div_y(gradFs_b[:,:,1])

        Is_r_init = np.zeros((img_r.shape[0], img_r.shape[1]))
        Is_g_init = np.zeros((img_r.shape[0], img_r.shape[1]))
        Is_b_init = np.zeros((img_r.shape[0], img_r.shape[1]))

        eps = 0.01
        N = 2000

        Is_r = self.poisson_solve(divFs_r, Is_r_init, eps, N, img_r)
        Is_g = self.poisson_solve(divFs_g, Is_g_init, eps, N, img_g)
        Is_b = self.poisson_solve(divFs_b, Is_b_init, eps, N, img_b)

        Is = np.dstack((Is_r, Is_g, Is_b))
        plt.imshow(Is)
        plt.show()
        return Is


        # diff = np.subtract(Is, img_rgb)
        # print(diff)
        # plt.imshow(diff)
        # plt.show()
        # print(Is)
        # fig = plt.figure()
        # fig.add_subplot(2,1,1)
        # plt.imshow(img)
        # fig.add_subplot(2,1,2)
        # plt.imshow(Is)
        # plt.show()
