import numpy as np
import matplotlib as plt
import pandas as pd

#Inspired by: https://christinakouridi.blog/2019/07/09/vanilla-gan-numpy/
#https://towardsdatascience.com/the-math-behind-gans-generative-adversarial-networks-3828f3469d9c

data = pd.read_csv('train.csv')

data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data = data.T

x_data = (data[1:] -127.5) / 127.5
y_data = data[0] 

number = 1
batch_size = 100

y_train = []
x_train = []
for i in range(y_data.shape[0]):
    if y_data[i] == number:
        y_train.append(y_data[i])
        x_train.append(x_data[:, i])

y_train = np.array(y_train)
x_train = np.array(x_train)

num_batches = x_train.shape[0] // batch_size
x_train = x_train[:num_batches * batch_size]
y_train = y_train[:num_batches * batch_size]


W1_g = np.random.rand(128,100) - 0.5
b1_g = np.random.rand(100,1) - 0.5
W2_g = np.random.rand(784,128) - 0.5
b2_g = np.random.rand(784,1) - 0.5

W1_d = np.random.rand(128,784) - 0.5
b1_d = np.random.rand(128,1) - 0.5
W2_d = np.random.rand(1,1) - 0.5
b2_d = np.random.rand(1,1) - 0.5

def lrelu(x, alpha = 0.01):
    return np.maximum(x, x * alpha)

def sigmoid(x):
    return 1/(1+np.exp(x))

def dlrelu(x, alpha= 0.01):
    dx = np.ones_like(x)
    dx[x<0] = alpha
    return dx

def dsigmoid(x):
    return x(1-x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2

def forward_gen(x):
    pass

def forward_discrim(x):
    pass

print(x_data.shape)