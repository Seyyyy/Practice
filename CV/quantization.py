import numpy as np
import cv2
import matplotlib.pyplot as plt


class Quantization:
    '''
    kernelのrowとcolは画像サイズを割り切れる値じゃないとだめ
    '''
    def __init__(self, kernel_r, kernel_c):
        self.kernel_r = kernel_r
        self.kernel_c = kernel_c


    def quantize2D(self, img):
        k_r, k_c = (int(img.shape[0]/self.kernel_r), int(img.shape[1]/self.kernel_c))
        kernel = np.full((k_r, k_c), 1 / (k_r * k_c))
        out = []
        for row in range(0, img.shape[0], k_r) :
            q_row = []
            for col in range(0, img.shape[1], k_c) :
                print(row)
                print(col)
                temp = np.sum(img[row : row + k_r, col : col + k_c] * kernel)
                q_row.append(temp)
            out.append(q_row)
        return out


img = cv2.imread('sample/IMG_5719.JPG', cv2.IMREAD_GRAYSCALE) # 1785, 1200
# img = cv2.imread('sample/IMG_5905.JPG', cv2.IMREAD_GRAYSCALE) # 2048, 2048
print(img.shape)
dst = cv2.Canny(img, 100, 200)
# dst = cv2.resize(img, None, fx=0.5, fy=1.0)

Q = Quantization(3, 3)
out = Q.quantize2D(dst)

print(out)
# print(dst)
plt.imshow(dst)
plt.show()