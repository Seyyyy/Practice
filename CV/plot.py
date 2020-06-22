import numpy as np
import matplotlib.pyplot as plt
import csv

Xy = []
with open('csv/color.csv') as fp:
    for row in csv.reader(fp, delimiter=','):
        Xy.append(row)

Xy = np.array(Xy[1:], dtype=np.float64)

x = Xy[:, 0]
y = np.arange(0, 1, 0.03125)
title=['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7','h8', 'h9', 'h10', 'h11']

fig, axes = plt.subplots(4, 3)

index = 0
for j in range(3):
    for i in range(4):
        x = Xy[:, i]
        axes[i][j].set_ylim([0, 1])
        axes[i][j].set_title(title[index])
        axes[i][j].scatter(x, y, color='r')
        index += 1

plt.show()
