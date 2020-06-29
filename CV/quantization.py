import numpy as np
import cv2
import matplotlib.pyplot as plt


class Quantization:
    '''
    kernelのrowとcolは画像サイズを割り切れる値じゃないとだめ
    '''
    def __init__(self, kernel_r, kernel_c, imgSrc):
        self.kernel_r = kernel_r
        self.kernel_c = kernel_c
        # 今のところは3分割を想定
        self.imgSrc = cv2.resize(imgSrc, (1785, 1200))
        self.quantizeImg = []
        self.avarage = []


    def quantize(self):
        k_r, k_c = (int(self.imgSrc.shape[1]/self.kernel_r), int(self.imgSrc.shape[0]/self.kernel_c))
        for row in range(0, self.imgSrc.shape[0] - k_r, k_r):
            print(row)
            for col in range(0, self.imgSrc.shape[1] - k_c, k_c):
                self.quantizeImg.append(self.imgSrc[row : row + k_r, col : col + k_c])
        self.quantizeImg = np.array(self.quantizeImg)


    def normalize(self):
        self.quantizeImg = self.quantizeImg / 255
        self.quantizeImg = np.where(self.quantizeImg > 0, 1, 0)


    def avaraging(self):
        for i in range(self.quantizeImg.shape[0]):
            self.avarage.append(np.mean(self.quantizeImg[i]))
        self.avarage = np.array(self.avarage)


    def quantize2D(self, imgSrc):
        img = cv2.resize(imgSrc, (1785, 1200))
        k_r, k_c = (int(img.shape[0]/self.kernel_r), int(img.shape[1]/self.kernel_c))
        kernel = np.full((k_r, k_c), 1 / (k_r * k_c))
        out = []
        for row in range(0, img.shape[0] - k_r, k_r) :
            q_row = []
            for col in range(0, img.shape[1] - k_c, k_c) :
                temp = np.sum(img[row : row + k_r, col : col + k_c] * kernel)
                q_row.append(temp)
            out.append(q_row)
        return out


# img = cv2.imread('sample/IMG_5719.JPG', cv2.IMREAD_GRAYSCALE) # 1785, 1200
img = cv2.imread('sample/IMG_5905.JPG', cv2.IMREAD_GRAYSCALE) # 2048, 2048
dst = cv2.Canny(img, 100, 200)

Q = Quantization(3, 3, dst)
# out = Q.quantize2D(dst)
Q.quantize()
Q.normalize()
Q.avaraging()
print(Q.avarage)
print(Q.quantizeImg.shape)
print(Q.imgSrc.shape)
# cv2.imwrite('out/out.jpg', Q.quantizeImg[2])
# plt.imshow(Q.quantizeImg[2])
# plt.show()