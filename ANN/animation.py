'''
@author: Ayesha Siddika Nipu (an37s)
Description: This program visualizes MSE values as a training curve.
'''

import ann_test as ANN
import matplotlib.pyplot as plt

fig = plt.figure(num="MSE Variation")

y = ANN.training_mse
ax = plt.axes(xlim=[0, len(y)], ylim=[min(y) - 0.01, max(y) + 0.01])

i = 0
for point in ANN.training_mse:
    ax.scatter(i, point, facecolor='blue')
    i += 1

j = 0
for point in ANN.validation_mse:
    ax.scatter(j, point, facecolor='green')
    j += 1

plt.show()
