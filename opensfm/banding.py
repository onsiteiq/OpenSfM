from __future__ import print_function
from __future__ import division
import os
import subprocess
import shutil
import logging
import glob
import cv2 as cv
import numpy as np
import ffmpeg
import pywt
import argparse

from opensfm import csfm

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.realpath(__file__))

alpha = 1.3
alpha_max = 500
beta = 40
beta_max = 200
gamma = 0.5
gamma_max = 200


def remove_banding(config, oiq_proc_dir):
    raw_dir = os.path.join(oiq_proc_dir, "RAW IMAGES")
    nctech_dir = os.path.join(oiq_proc_dir, "nctech_imu")

    # find raw mkv files input
    valid_dir = raw_dir
    mkv_files = glob.glob(os.path.join(raw_dir, '*.mkv'))

    if not mkv_files:
        valid_dir = nctech_dir
        mkv_files = glob.glob(os.path.join(nctech_dir, '*.mkv'))

        if not mkv_files:
            valid_dir = oiq_proc_dir
            mkv_files = glob.glob(os.path.join(oiq_proc_dir, '*.mkv'))

    if mkv_files:
        os.chdir(valid_dir)

        for mkv_file in mkv_files:
            # step 1 - use ffmpeg to extract frames from mkv file
            #subprocess.call(['ffmpeg', '-i', mkv_file, 'img%04d.jpg', '-codec', 'copy'])
            ffmpeg.input(mkv_file).output('img%04d*.jpg').run()

            # step 2 - call horizontal_banding_removal for each frame
            img_files = sorted(glob.glob(os.path.join(valid_dir, 'img*.jpg')))
            for img_file in img_files:
                img_original = cv.imread(img_file)
                if csfm.is_banding_present(img_file):
                    #csfm.run_notch_filter()
                    ##img_corrected = gamma_correction(img_original)
                    #img_corrected = basic_linear_transform(img_original)
                    img_corrected = horizontal_banding_removal(img_original)

                    cv.imwrite(img_file, img_corrected)

            # step 3 - save original mkv. use ffmpeg to encode processed frames to new mkv.
            mkv_file_orig = mkv_file + ".orig"
            shutil.move(mkv_file, mkv_file_orig)

            #subprocess.call(['ffmpeg', '-i', 'img%04d.jpg', '-r', '7', '-codec', 'copy', mkv_file])
            ffmpeg.input('img%04d*.jpg').output(mkv_file).run()

            # step 4 - clean up
            for img_file in img_files:
                os.remove(img_file)


def basic_linear_transform(img_original):
    return cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
 

def gamma_correction(img_original):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(img_original, lookUpTable)


def horizontal_banding_removal(image_original, level=None, wname='db5', sigma=5.0, pad=True):
    '''
    implements algorithm described in "Stripe and ring artifact removal with combined
    wavelet — Fourier filtering"
    :param image_original:
    :param level:
    :param wname:
    :param sigma:
    :param pad:
    :return:
    '''
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
    parser = argparse.ArgumentParser(description='correct banding')
    parser.add_argument('--input', help='Path to input images.', default='./')
    args = parser.parse_args()

    img_original = cv.imread(args.input)
    if csfm.is_banding_present(args.input):
        # csfm.run_notch_filter()
        ##img_corrected = gamma_correction(img_original)
        # img_corrected = basic_linear_transform(img_original)
        img_corrected = horizontal_banding_removal(img_original)

        cv.imwrite(args.input + ".new.jpg", img_corrected)
