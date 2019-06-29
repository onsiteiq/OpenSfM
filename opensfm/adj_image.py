from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

alpha = 1.3
alpha_max = 500
beta = 40
beta_max = 200
gamma = 0.5
gamma_max = 200

def basicLinearTransform():
    return cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
 

def gammaCorrection():
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(img_original, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()

img_original = cv.imread(cv.samples.findFile(args.input))
if img_original is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

#img_corrected = gammaCorrection()
img_corrected = basicLinearTransform()

output_filename = args.input + ".new.jpg"
cv.imwrite(output_filename, img_corrected)
