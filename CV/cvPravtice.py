import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import math


def plotOne(img, nrows, ncols, index):
    '''グリッドレイアウトで並べるための１要素'''
    plt.subplot(nrows, ncols, index)
    plt.imshow(img, cmap='Dark2')
    plt.axis('off')


def img2magnitudeSpectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum


def directory2plt():
    '''sampleディレクトリからスペクトルをプロット'''
    images = glob.glob('sample/*')
    index = len(images) * 2
    nrows = 4
    ncols = math.ceil(index / nrows)
    offset = 1
    for path in images:
        img = cv2.imread(path,0)
        magnitude_spectrum = img2magnitudeSpectrum(img)
        plotOne(magnitude_spectrum, nrows, ncols, offset)
        offset += 1
        plotOne(img, nrows, ncols, offset)
        offset += 1
    # plt.show()
    plt.savefig('plt.png', dpi=800)


def img2plt():
    img = cv2.imread('sample/IMG_5767.JPG',0)
    magnitude_spectrum = img2magnitudeSpectrum(img)
    plotOne(magnitude_spectrum, 1, 2, 1)
    plotOne(img, 1, 2, 2)
    # plt.show()
    plt.savefig('pltimg.png', dpi=300)

img2plt()