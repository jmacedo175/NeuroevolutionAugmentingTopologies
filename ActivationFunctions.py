import numpy as np
def logistics(x):
    return 1 / (1 + np.exp(-x))

def sinusoid(x):
    return np.sin(x)

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def leaky_relu(x):
    return np.maximum(0.01*x, x)

def binary_step(x):
    return 0 if x < 0 else 1

def arctan(x):
    return np.arctan(x)

def elu(x):
    return x if x > 0 else np.exp(x) - 1

def softplus(x):
    return np.log(1 + np.exp(x))

def modified_sigmoid(x):
    #function used in the NEAT paper
    return 1/(1+np.exp(-4.9*x))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(-10,10,100)
    plt.subplot (2,3,1) 
    plt.title('Logistics')
    plt.plot(x,logistics(x))
    plt.subplot (2,3,2)
    plt.title('Sinusoid')
    plt.plot(x,sinusoid(x))
    plt.subplot (2,3,3)
    plt.title('ReLU')
    plt.plot(x,relu(x))
    plt.subplot (2,3,4)
    plt.title('Tanh')
    plt.plot(x,tanh(x))
    plt.subplot (2,3,5)
    plt.title('Linear')
    plt.plot(x,linear(x))
    
    plt.show()
    