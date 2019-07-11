from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import pywt
import argparse

from opensfm import csfm

alpha = 1.3
alpha_max = 500
beta = 40
beta_max = 200
gamma = 0.5
gamma_max = 200


def basic_linear_transform():
    return cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
 

def gamma_correction():
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(img_original, lookUpTable)


def remove_stripe_fw(image_original, level=None, wname='db5', sigma=2, pad=True):
    image_rotated = cv.rotate(image_original, cv.ROTATE_90_CLOCKWISE)

    dx, dy, dz = image_rotated.shape
    nx = dx
    if pad:
        nx = dx + dx // 8
    xshift = int((nx - dx) // 2)

    if level is None:
        size = np.max(image_rotated.shape)
        level = int(np.ceil(np.log2(size)))

    for m in range(dz):
        sli = np.zeros((nx, dy), dtype='float32')
        sli[xshift:dx + xshift] = image_rotated[:, :, m]

        # Wavelet decomposition.
        cH = []
        cV = []
        cD = []
        for n in range(level):
            sli, (cHt, cVt, cDt) = pywt.dwt2(sli, wname)
            cH.append(cHt)
            cV.append(cVt)
            cD.append(cDt)

        # FFT transform of horizontal frequency bands.
        for n in range(level):
            # FFT
            fcV = np.fft.fftshift(np.fft.fft(cV[n], axis=0))
            my, mx = fcV.shape

            # Damping of ring artifact information.
            y_hat = (np.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -np.expm1(-np.square(y_hat) / (2 * np.square(sigma)))
            fcV *= np.transpose(np.tile(damp, (mx, 1)))

            # Inverse FFT.
            cV[n] = np.real(np.fft.ifft(np.fft.ifftshift(fcV), axis=0))

        # Wavelet reconstruction.
        for n in range(level)[::-1]:
            sli = sli[0:cH[n].shape[0], 0:cH[n].shape[1]]
            sli = pywt.idwt2((sli, (cH[n], cV[n], cD[n])), wname)

        image_rotated[:, :, m] = sli[xshift:dx + xshift, 0:dy]

    return cv.rotate(image_rotated, cv.ROTATE_90_COUNTERCLOCKWISE)


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness')
    parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
    args = parser.parse_args()

    img_original = cv.imread(args.input)
    if img_original is None:
        print('Could not open or find the image: ', args.input)
        exit(0)

    if csfm.is_banding_present(args.input):
        #csfm.run_notch_filter()
        ##img_corrected = gamma_correction()
        #img_corrected = basic_linear_transform()
        img_corrected = remove_stripe_fw(img_original)

        output_filename = args.input + ".new.jpg"
        cv.imwrite(output_filename, img_corrected)



