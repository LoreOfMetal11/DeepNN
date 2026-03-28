import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv", header=None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values
y = np.where(y == 'Iris-setosa', 0, 1)  
y = y.astype(np.longlong)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

linear = nn.Linear(4, 2)
lossFn = nn.CrossEntropyLoss()
optimizer = optim.SGD(linear.parameters(), lr=0.01)
progoni = 101
for progon in range(progoni):
    optimizer.zero_grad()
    outputs = linear(X_train)
    loss = lossFn(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Прогон {progon + 1}/{progoni}, ошибка: {loss.item():.4f}')

with torch.no_grad():
    outputs = linear(X_test)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test).sum().item()
    total = len(y_test)
    accuracy = 100 * correct / total
    print(f'Точность: {accuracy:.2f}%')
    
