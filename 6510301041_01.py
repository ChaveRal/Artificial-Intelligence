"""
Stud ID: 6510301041
Name   : Cheewapron Sutus
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Raw Data
x = np.array([29, 28, 34, 31, 25]).reshape((-1, 1)) #High
y = np.array([77, 62, 93, 84, 59]) #weigh

model = LinearRegression().fit(x, y)

#calculate a, b
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq:.2f}")
print(f"intercept: {model.intercept_:.2f}")
print(f"slope: {model.coef_[0]:.2f}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

# create graph
plt.figure(figsize=(8, 6))

# point data
plt.scatter(x, y, color='blue', label='Raw Data')  # จุดข้อมูลจริง

#Linear Regression line
plt.plot(x, y_pred, color='red', linestyle='--', label='Regression Line')  # เส้น Linear Regression

# scale x and y 
x_margin = 1  # High
y_margin = 5  # Weigh

#schedul scales
plt.xlim(min(x)[0] - x_margin, max(x)[0] + x_margin)
plt.ylim(min(y) - y_margin, max(y) + y_margin)

# Drawing axes
plt.axhline(0, linestyle='-.', color='magenta', linewidth=1)
plt.axvline(0, linestyle='-.', color='magenta', linewidth=1)

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Linear Regression: coeff={r_sq:.2f}')
plt.legend()
plt.grid()
plt.tight_layout(pad=0)
plt.show()


