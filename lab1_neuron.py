import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2, 3]].values

def neuron(w, x):
    return 1 if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0 else -1

w = np.random.random(4)
eta = 0.01
w_iter = []
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    w[1:] += eta * (target - predict) * xi
    w[0] += eta * (target - predict)
    w_iter.append(w.tolist())

sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    sum_err += (target - predict) / 2

print("Всего ошибок:", sum_err)

xl = np.linspace(min(X[:, 0]), max(X[:, 0]))
plt.figure()
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x')

for i, w in enumerate(w_iter):
    yl = -(xl * w[1] + w[0]) / w[2] 
    plt.plot(xl, yl)
    plt.text(xl[-1], yl[-1], i, dict(size=10, color='gray'))
    plt.pause(0.1)

plt.text(xl[-1] - 0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.show()
