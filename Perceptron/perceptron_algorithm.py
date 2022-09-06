'''
@author: Ayesha Siddika Nipu (an37s)
Description: This program implements Perceptron Learning Algorithm
'''

from random import random, randint, uniform
import math

# defines slope range
m = uniform(math.pi / 6, math.pi / 3)
c = randint(-50, 50)

epochs = 0
learning_rate = 0.0001
data_points = 20
dataset = []
points = []

# weight initialization
weights = [random(), random(), random()]

# Generate dataset
for i in range(data_points):
    x = randint(100, 900)
    y = randint(100, 900)
    pt_class = 1.0 if y - m * x - c >= 0.00 else -1.0
    dataset.append([1, x, y, pt_class])


def activate(s):
    return 1.0 if s >= 0.0 else -1.0


def feed_forward(row):
    sigma = 0
    for weight, x in zip(weights, row):
        sigma += weight * x

    return activate(sigma)


def train_data(row):
    output = feed_forward(row[0:3])
    for i in range(len(row) - 1):
        weights[i] += learning_rate * (row[-1] - output) * row[i]


def get_current_points(points):
    x_0 = 0
    x_n = 1000
    y_0 = - weights[0] / weights[2]
    y_n = - (weights[1] * 1000 + weights[0]) / weights[2]

    points.append([[x_0, x_n], [y_0, y_n]])


while (True):
    misclassified = 0
    converged = True
    epochs += 1
    for curr_row in dataset:
        classify = feed_forward(curr_row[0:3])
        train_data(curr_row)
        if (curr_row[-1] != classify):
            misclassified += 1
            converged = False

    get_current_points(points)
    print("Epoch: ", epochs, "\t", "Number of Misclassification: ", misclassified)
    print()
    if (converged):
        break
