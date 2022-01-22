import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from pathlib import Path

data = pd.read_csv('train.csv')

data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data = data.T

x_data = (data[1:] -127.5) / 127.5
y_data = data[0] 

create_gif = True
image_dir = Path('./GAN_Images')
number = 1
batch_size = 100

if not image_dir.is_dir():
    image_dir.mkdir()

filenames = []

#Getting dataset of only 1 number
y_train = []
x_train = []
for i in range(y_data.shape[0]):
    if y_data[i] == number:
        y_train.append(y_data[i])
        x_train.append(x_data[:, i])

y_train = np.array(y_train)
x_train = np.array(x_train)

#Getting rid of partial batch
num_batches = x_train.shape[0] // batch_size
x_train = x_train[:num_batches * batch_size]
y_train = y_train[:num_batches * batch_size]

alpha = 0.1

#Initializing weights/biases 
#100 -> 128 -> 784
W1_g = np.random.rand(128,100) - 0.5
b1_g = np.random.rand(100,1) - 0.5
W2_g = np.random.rand(784,128) - 0.5
b2_g = np.random.rand(784,1) - 0.5

#784 -> 128 -> 1
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

#aa
def forward_gen(x):
    preL1_g = np.dot(x, W1_g) + b1_g
    L1_g = lrelu(preL1_g, alpha = 0)

    preL2_g = np.dot(L1_g, W2_g) + b2_g
    L2_g = np.tanh(preL2_g)
    return preL2_g, L2_g, L1_g, preL1_g    

def forward_discrim(x):
    preL1_d = np.dot(x, W1_d) + b1_d
    L1_d = lrelu(preL1_d)

    preL2_d = np.dot(L1_d, W2_d) + b2_d
    L2_d = sigmoid(preL2_d)
    return preL2_d, L2_d, L1_d, preL1_d

def backward_discrim(x_real, preL1_real, L1_real, x_fake, preL1_fake, L1_fake, L1_d, preL1_d):
    dL2_real = -1 / (L1_real + 1e-8)
    dpreL2_real = dL2_real * dsigmoid(preL1_real)
    dW2_real = np.dot(L1_d.T,dpreL2_real)
    db2_real = np.sum(dpreL2_real, axis=0, keepdims=True)
    
    dL1_real = np.dot(dpreL2_real, W1_d.T)
    dpreL1_real = dL1_real * dlrelu(preL1_d)
    dW1_real = np.dot(x_real.T, dpreL1_real)
    db1_real = np.sum(dpreL1_real, axis=0, keepdims=True)

    dL2_fake = 1 / (1 - L1_real + 1e-8)
    dpreL2_fake = dL2_fake * dsigmoid(preL1_fake)
    dW2_fake = np.dot(L1_d.T, dpreL2_fake)
    db2_fake = np.sum(dpreL2_fake, axis=0, keepdims=True)

    dL1_fake = np.dot(dpreL2_fake, W1_d.T)
    dpreL1_fake = dL1_fake * dlrelu(preL1_d, alpha=0)
    dW1_fake = np.dot(x_fake.T, dpreL1_fake)
    db1_fake = np.sum(dpreL1_fake, axis=0, keepdims=True)

    dW2 = dW2_real + dW2_fake
    db2 = db2_real + db2_fake

    dW1 = dW1_real + dW1_fake
    db1 = db1_real + db1_fake

    W2_d -= alpha * dW2
    b2_d -= alpha * db2

    W1_d -= alpha * dW1
    b1_d -= alpha * db1

def backward_gen(x, preL1_fake, L1_fake, L2_fake, preL1_d, L1_g, preL1_g):
    dL2_d = -1 / (L2_fake + 1e-8)

    dpreL2_d = dL2_d * dsigmoid(L1_fake)
    dL1_d = np.dot(dpreL2_d, W1_d.T)
    dW1_d = dL1_d * dlrelu(preL1_d)
    dL2_d = np.dot(dW1_d, W1_d.T)

    dpreL2_g = dL2_d * dtanh(L1_g)
    dW2_g = np.dot(dpreL2_g, L1_g)
    db2_g = np.sum(dpreL2_g, axis = 0, keepdims=True)

    dL1_g = np.dot(dpreL2_g, W1_g.T)
    dpreL1_g = dL1_g * dlrelu(preL1_g, alpha=0)
    dW1_g = np.dot(dpreL1_g, x.T)
    db1_g = np.sum(dpreL1_g, axis=0, keepdims=True)

    W1_g -= alpha * dW1_g
    b1_g -= alpha * db1_g

    W2_g -= alpha * dW2_g
    b2_g -= alpha * db2_g

def sample_images(images, epoch, show):
    images = np.reshape(images, (batch_size, 28, 28))

    fig = plt.figure(figsize=(4,4))

    if create_gif:
        current_epoch_filename = image_dir.joinpath(f"GAN_epoch{epoch}.png")
        filenames.append(current_epoch_filename)
        plt.savefig(current_epoch_filename)

    if show == True:
        plt.show()
    else:
        plt.close()

def generate_gif():
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave("GAN.gif", images)

