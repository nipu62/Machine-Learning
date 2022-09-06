'''
Assignment 2: Linear Regression
@author: Ayesha Siddika Nipu (an37s)
'''

from random import randint

w = randint(1, 5)
b = randint(1, 10)

x_val = []
y_val = []

total = 20
epochs = 50
learning_rate = 0.0001

for i in range(total):
    x_val.append(randint(1, 50))
    y = w * x_val[i] + b
    y_val.append(y + (-1) ** randint(0, 1) * randint(0, round(0.1 * y)))

def calculate_errors(w, b):
    err_b = 0
    err_w = 0
    mean_squarred_error = 0
    for i in range(total):
        y = (w * x_val[i] + b)
        err_w += x_val[i] * (y_val[i] - y)
        err_b += y_val[i] - y
        mean_squarred_error += ((y_val[i] - y) ** 2) / total

    return err_w, err_b, mean_squarred_error
