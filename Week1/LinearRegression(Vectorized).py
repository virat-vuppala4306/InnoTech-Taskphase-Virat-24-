import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_set = pd.read_csv("Salary Data.csv")

x = training_set["YearsExperience"].values
y = training_set["Salary"].values

def cost_function(x, y, w, b):
    m = len(x)
    y_hat = w * x + b
    return (1 / m) * np.sum((y_hat - y) ** 2)


def gradient_function(x, y, w, b):
    m = len(x)

    y_hat = w * x + b
    error = y_hat - y

    dw = (2 / m) * np.dot(error, x)
    db = (2 / m) * np.sum(error)

    return dw, db

def gradient_descent(x, y, alpha, iterations):
    w, b = 0.0, 0.0

    for _ in range(iterations):
        dw, db = gradient_function(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db

    return w, b


learning_rate = 0.01
iterations = 10000

final_w, final_b = gradient_descent(x, y, learning_rate, iterations)

print(f"w: {final_w:.4f}, b: {final_b:.4f}")


plt.scatter(x, y, label="Data Points")

x_vals = np.linspace(min(x), max(x), 100)
y_vals = final_w * x_vals + final_b

plt.plot(x_vals, y_vals, color="red", label="Regression Line")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.show()

import time

w, b = 0.0, 0.0

x_big = np.tile(x, 100000)
y_big = np.tile(y, 100000)

# Vectorized version timing
start = time.time()
dw2, db2 = gradient_function(x_big, y_big, w, b)
vector_time = time.time() - start

print(f"Vectorized gradient time: {vector_time:.4f} seconds")
