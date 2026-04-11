import torch 
import torch.nn as nn 
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('dataset_simple.csv')
X = torch.Tensor(df.iloc[:, [0]].values)
y = torch.Tensor(df.iloc[:, [1]].values)
class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        
    def forward(self, X):
        pred = self.layers(X)
        return pred

inputSize = X.shape[1]
hiddenSizes = 3
outputSize = 1

net = NNet_regression(inputSize, hiddenSizes, outputSize)
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.000000001)
epohs = 5000
for i in range(0, epohs):
    pred = net.forward(X)
    loss = lossFn(pred.squeeze(), y.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

with torch.no_grad():
    pred = net.forward(X)

print('\nПредсказания:')
print(pred[0:10])

err = torch.mean(abs(y - pred))
print('\nОшибка: ')
print(err)
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, [1]].values, marker='o')

with torch.no_grad():
    y1 = net.forward(torch.Tensor([[20.0]]))
    y2 = net.forward(torch.Tensor([[60.0]]))

plt.plot([20, 60], [y1.numpy()[0][0], y2.numpy()[0][0]], 'r')
plt.xlabel('age')
plt.ylabel('income')
plt.grid(True)
plt.show()