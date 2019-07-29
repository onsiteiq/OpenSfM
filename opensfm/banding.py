from __future__ import print_function
from __future__ import division
import os
import shutil
import ntpath
import logging
import glob
import cv2 as cv
import numpy as np
import ffmpeg
import pywt
import argparse

from opensfm import csfm
from opensfm import log
from opensfm.context import parallel_map

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.realpath(__file__))

alpha = 1.3
alpha_max = 500
beta = 40
beta_max = 200
gamma = 0.5
gamma_max = 200


def remove_banding( num_processes, mkv_search_paths = [], working_dir = None):
    
    if working_dir is None:
        working_dir = os.getcwd()

    # find raw mkv files input

    valid_dir = None
    for mkv_path in mkv_search_paths:
        
        mkv_files = glob.glob( os.path.join(mkv_path, '*.mkv') )
        if mkv_files:
            valid_dir = mkv_path
            break
    
    if valid_dir is None:
        valid_dir = working_dir
        mkv_files = glob.glob(os.path.join(working_dir, '*.mkv'))

    if mkv_files and len(mkv_files) == 4:
        mkv_dirs = []

        # use ffmpeg to extract frames from mkv file
        for mkv_file in mkv_files:
            os.chdir(valid_dir)
            mkv_dir = os.path.splitext(mkv_file)[0]
            mkv_dirs.append(mkv_dir)
            os.makedirs(mkv_dir, exist_ok=True)
            os.chdir(mkv_dir)

            #subprocess.call(['ffmpeg', '-i', mkv_file, 'img%04d.jpg', '-codec', 'copy'])
            ffmpeg.input(mkv_file).output('%04d.jpg').run()

        # call horizontal_banding_removal for each set of 4 frames
        args = [(idx, mkv_dirs) for idx in range(1, len(glob.glob(os.path.join(mkv_dirs[0], '*.jpg')))+1)]
        parallel_map(remove, args, num_processes)

        # save original mkv. use ffmpeg to encode processed frames to new mkv.
        for mkv_file in mkv_files:
            os.chdir(valid_dir)
            mkv_file_orig = mkv_file + ".orig"
            shutil.move(mkv_file, mkv_file_orig)

        for mkv_dir in mkv_dirs:
            os.chdir(mkv_dir)

            #subprocess.call(['ffmpeg', '-i', 'img%04d.jpg', '-r', '7', '-codec', 'copy', mkv_file])
            ffmpeg.input('%04d.jpg').output(mkv_dir + ".mkv").run()
            
            os.chdir(working_dir)
            
            shutil.rmtree(mkv_dir)

    os.chdir(working_dir)


def remove(args):
    log.setup()

    idx, mkv_dirs = args

    # TESTING - only correct 10 images
    #if idx > 10:
        #return

    img_set = []
    for mkv_dir in mkv_dirs:
        img_set.append(os.path.join(mkv_dir, _int_to_shot_id(idx)))

    banding_cnt = 0
    for img_file in img_set:
        freq = csfm.is_banding_present(img_file)
        if freq != -1:
            logger.info('{}: {}Hz banding'.format(idx, freq))
            banding_cnt += 1
        else:
            logger.info('{}: no banding'.format(idx))

    logger.info('{}: banding found in {}/4 images'.format(_int_to_shot_id(idx), banding_cnt))

    if banding_cnt >= 1:
        logger.info('{}: removing banding...'.format(idx))

        for img_file in img_set:
            img_original = cv.imread(img_file)
            #csfm.run_notch_filter()
            ##img_corrected = gamma_correction(img_original)
            #img_corrected = basic_linear_transform(img_original)
            img_corrected = horizontal_banding_removal(img_original)

            # high intensity pixels' color is distorted by the filtering process above.
            # so we replace them by their original values
            grayscaled = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
            retval, mask = cv.threshold(grayscaled, 220, 255, cv.THRESH_BINARY)
            img_corrected[mask == 255] = img_original[mask == 255]

            # there is a 72x64 pixel area in each raw image that appears to be 2D barcode
            # used by the NC Tech Immersive Studio. don't know if the size of this area
            # changes when image resolution changes (currently 3968x3008), so need to keep
            # an eye on it if/when we switch camera resolution
            img_corrected[0:64, 0:72] = img_original[0:64, 0:72]

            cv.imwrite(img_file, img_corrected)


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


def _shot_id_to_int(shot_id):
    """
    Returns: shot id to integer
    """
    tokens = shot_id.split(".")
    return int(tokens[0])


def _int_to_shot_id(shot_int):
    """
    Returns: integer to shot id
    """
    return str(shot_int).zfill(4) + ".jpg"


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correct banding')
    parser.add_argument('--input', help='Path to input images.', default='./')
    args = parser.parse_args()

    img = cv.imread(args.input)
    if csfm.is_banding_present(args.input):
        # csfm.run_notch_filter()
        ##img_processed = gamma_correction(img)
        # img_processed = basic_linear_transform(img)
        img_processed = horizontal_banding_removal(img)

        grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        retval, mask = cv.threshold(grayscaled, 220, 255, cv.THRESH_BINARY)
        img_processed[mask == 255] = img[mask == 255]
        img_processed[0:64, 0:72] = img[0:64, 0:72]

        cv.imwrite(args.input + ".new.jpg", img_processed)
