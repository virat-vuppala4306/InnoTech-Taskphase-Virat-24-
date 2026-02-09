import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_set = pd.read_csv("Salary Data.csv")

x_values = training_set["YearsExperience"].values
y_values = training_set["Salary"].values

plt.scatter(x_values, y_values)
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()
#Visualise the dataset

#Defining the cost function for the linear regression model 
def cost_function(x,y,w,b):
    m = len(x)
    cost_sum = 0

    for i in range(m):
        f = w * x[i] + b
        cost = (f - y[i]) ** 2
        cost_sum += cost
    
    total_cost = (1/m) * cost_sum
    return total_cost

#Defining function to find the derivatives for w and b
def gradient_function(x, y, w, b):
    m = len(x)
    dc_dw = 0
    dc_db = 0

    for i in range(m):
        f = w * x[i] + b

        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])

    dc_dw = (2/m) * dc_dw
    dc_db = (2/m) * dc_db

    return dc_dw, dc_db

#Gradient descent function to implement the above results
def gradient_descent(x, y, alpha, iterations):
    w = 0
    b = 0

    for i in range(iterations):
        dc_dw, dc_db = gradient_function(x, y, w, b)

        w = w - alpha * dc_dw
        b = b - alpha * dc_db

    return w, b

learning_rate = 0.01
iterations = 10000

final_w, final_b = gradient_descent(x_values, y_values, learning_rate, iterations)

print(f"w: {final_w:.4f}, b: {final_b:.4f}")

#Implementing everything and finding values for w and b after 10000 iterations

plt.scatter(x_values, y_values, label='Data Points')

x_vals = np.linspace(min(x_values), max(x_values), 100)
y_vals = final_w * x_vals + final_b
plt.plot(x_vals, y_vals, color='red', label='Regression Line')

plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.show()

#Visualising the line achieved by the model to predict values

import time

w, b = 0.0, 0.0

x_big = np.tile(x_values, 100000)
y_big = np.tile(y_values, 100000)

# Loop version timing
start = time.time()
dw1, db1 = gradient_function(x_big, y_big, w, b)
loop_time = time.time() - start

print(f"Loop gradient time: {loop_time:.4f} seconds")