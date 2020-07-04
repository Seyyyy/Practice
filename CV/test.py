import numpy as np
import matplotlib as plt
import cv2
import math
import csv
# ----------------------------------------------------------
# img = np.full((1200, 1200, 3), [0, 0, 255], dtype=np.uint8)
# hue = 0
# saturation = 0
# for i in range(0, 1200, 48):
#     value = 0
#     for j in range(0, 1200, 48):
#         img[j: j+48, i: i+48, :] = [hue, saturation, value]
#         if(saturation * value <= 3000):
#             img[j: j+48, i: i+48, :] = [80, 255, 255]
#         value += 10
#     saturation +=10
# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
# cv2.imwrite('out/out2.jpg', img)
# ----------------------------------------------------------

# ----------------------------------------------------------
img = cv2.imread('sample/IMG_5463.JPG', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# print(img[img[:,:,1] < 40])
# print(img.shape)
# print(img[img[:,:,1] < 40].shape)
# print(img[img[:,:,1] >= 40].shape)

red = [0, 200, 200]
green = [30, 200, 200]
grey = [181, 45, 256]
black = [181, 256, 70]
# img[np.where((img < black).all(axis=2))] = green
# img[np.where((img < grey).all(axis=2))] = red

# a1 = 0
# for i in img:
#     a2 = 0
#     for j in i:
#         a = int(img[a1, a2, 1])
#         b = int(img[a1, a2, 2])
#         s = a * b
#         if(s <= 3000):
#             img[a1, a2] = [0, 200, 200]
#         a2 += 1
#     a1 += 1

img[np.where((img[:, :, 1].astype(np.int) * img[:, :, 2].astype(np.int)) < 3000)] = [80, 255, 255]
img[np.where(img[:, :, 0] < 15)] = [0, 255, 255]
# print(img)
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imwrite('out/out2.jpg', img)

# plt.imshow(img)
# plt.show()
# ----------------------------------------------------------

# v * s = 1500 以下にしたらうまくいきそう
# img = np.full((1200, 1200, 3), [0, 0, 255], dtype=np.uint8)
# entropy = np.full((25, 25), np.zeros(25, dtype=np.float))
# hue = 30
# saturation = 0
# en_i = 0
# en_j = 0
# for i in range(0, 1200, 48):
#     value = 0
#     en_j = 0
#     for j in range(0, 1200, 48):
#         img[j: j+48, i: i+48, :] = [hue, saturation, value]
#         entropy[en_j, en_i] = saturation * value
#         value += 10
#         en_j += 1
#     saturation +=10
#     en_i += 1

# with open('csv/entropy.csv', 'w') as f:
#     writer = csv.writer(f)
#     for i in entropy:
#         writer.writerow(i)


# ----------------------------------------------------------
# img = np.full((1200, 1200, 3), [0, 0, 255], dtype=np.uint8)
# hue = 0
# saturation = 0
# for i in range(0, 1200, 48):
#     value = 0
#     for j in range(0, 1200, 48):
#         img[j: j+48, i: i+48, :] = [hue, saturation, value]
#         value += 10
#     saturation +=10

# img[np.where((img[:, :, 1].astype(np.int) * img[:, :, 2].astype(np.int)) > 3000)] = [80, 255, 255]

# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
# cv2.imwrite('out/out2.jpg', img)
# ----------------------------------------------------------