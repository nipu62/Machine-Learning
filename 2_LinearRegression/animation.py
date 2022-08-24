'''
Assignment 2: Linear Regression
@author: Ayesha Siddika Nipu (an37s)
'''

import linear_regression as lr
from random import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

figure = plt.figure(num = "Assignment 2: Linear Regression [Gradient Descent Algorithm]")

X_0 = min(lr.x_val)
Y_0 = X_0 * lr.w + lr.b

X_n = max(lr.x_val)
Y_n = X_n * lr.w + lr.b

w1 = random()
b1 = random()

Y_min = min(Y_0, min(lr.y_val), (X_n * w1 + b1))
Y_max = max(Y_n, max(lr.y_val))

axis = plt.axes(xlim = (X_0 - 5, X_n + 5), ylim = (Y_min - 5 , Y_max + 5))

axis.plot([X_0, X_n], [Y_0, Y_n], color = "green", lw = 3, zorder= 1)
axis.plot(lr.x_val, lr.y_val, 'ro', zorder = 2)

[line] = axis.plot([], [], lw=3, color = "orange", zorder = 3)

points = []

for i in range(lr.epochs):
    err_w, err_b, mse = lr.calculate_errors(w1, b1)
    w1 += (2/lr.total) * err_w * lr.learning_rate
    b1 += (2/lr.total) * err_b * lr.learning_rate

    print("Epoch:{0:4d} \t Mean Square Error: {1:6.5f}".format(i+1,mse))
    Yn = X_n * w1 + b1
    points.append([[X_0, X_n], [Y_0, Yn]])

def show_animation(i):
    if(i >= len(points)):
        plt.pause(1)
        plt.close('all')
    else:
        line.set_data(points[i][0], points[i][1])

animation = FuncAnimation(
        figure,
        show_animation,
        frames = lr.epochs + 1,
        interval = 100)

plt.show()
