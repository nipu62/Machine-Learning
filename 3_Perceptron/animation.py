'''
@author: Ayesha Siddika Nipu (an37s)
Description: This program visualizes the line represented by the current
weights at the end of each epoch on GUI.
'''

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import perceptron_algorithm as per_alg

figure = plt.figure(num="Perceptron Learning Algrotihm")
axis = plt.axes(xlim=[0, 1000], ylim=[0, 1000])
axis.plot([0, 1000], [per_alg.c, per_alg.m * 1000 + per_alg.c], color="green", lw=3, zorder=1)

for point in per_alg.dataset:
    if point[3] == 1:
        face_color = "k"
    else:
        face_color = "none"
    axis.scatter(point[1], point[2], facecolor=face_color, edgecolor='k', zorder=2)

[line] = axis.plot([], [], lw=3, color="red", zorder=1)


def draw_animate(i):
    print('start animation')
    l = len(per_alg.points)
    if (i >= l):
        plt.pause(1)
        plt.close('all')
    else:
        line.set_data(per_alg.points[i][0], per_alg.points[i][1])
    print('end animation')

animation = FuncAnimation(
    figure,
    draw_animate,
    frames=per_alg.epochs + 1,
    interval=250)

plt.show()
