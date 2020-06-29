import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import csv


class Quantization:
    '''
    kernelのrowとcolは画像サイズを割り切れる値じゃないとだめ
    '''
    def __init__(self, kernel_r, kernel_c, imgSrc):
        self.kernel_r = kernel_r
        self.kernel_c = kernel_c
        # 今のところは3分割を想定
        self.imgSrc = cv2.resize(imgSrc, (1500, 1500))
        self.quantizeImg = []
        self.avarage = []
        self.overallAvarage = []


    def quantize(self):
        k_r, k_c = (int(self.imgSrc.shape[1]/self.kernel_r), int(self.imgSrc.shape[0]/self.kernel_c))# resize(x, y)の場合shapeは(y, x)になる
        for row in range(0, self.imgSrc.shape[0], k_r):
            for col in range(0, self.imgSrc.shape[1], k_c):
                self.quantizeImg.append(self.imgSrc[row : row + k_r, col : col + k_c])
        self.quantizeImg = np.array(self.quantizeImg)


    def normalize(self):
        self.quantizeImg = self.quantizeImg / 255
        self.quantizeImg = np.where(self.quantizeImg > 0, 1, 0)


    def avaraging(self):
        for i in range(self.quantizeImg.shape[0]):
            self.avarage.append(np.mean(self.quantizeImg[i]))
        self.avarage = np.array(self.avarage)


    def overallAveraging(self):
        for i in range(self.avarage.shape[0]):
            self.overallAvarage.append(self.avarage[i] / np.sum(self.avarage))
        self.overallAvarage = np.array(self.overallAvarage)


def mainFunc(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(img, 100, 200)
    Q = Quantization(3, 3, dst)
    Q.quantize()
    Q.normalize()
    Q.avaraging()
    Q.overallAveraging()
    return np.array(Q.overallAvarage)


images = glob.glob('sample/*')
result = np.zeros((9,))
for path in images:
    result += mainFunc(path)

csvに書き込み
with open('csv/edge.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(result)

# x = np.arange(0, 9, 1)

# cv2.imwrite('out/out.jpg', Q.quantizeImg[2])
# plt.imshow(Q.quantizeImg[2])
# plt.show()