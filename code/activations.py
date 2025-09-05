import numpy as np

# Sigmoid activation function

def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) # [0, 1]

# Hyperbolic tangent activation function
def tanh(x):
    return np.tanh(x) # [-1, 1]

# Identity activation function
def identity(x):
    return x # [x]

# Sine activation function
def sin(x):
    return np.sin(x) # [-1, 1]

# ReLU activation function
def relu(x):
    return np.maximum(0, x) # [0, x]

# Cosine activation function
def cos(x):
    return np.cos(x) # [-1, 1]

# Gaussian activation function
def gaussian(x):
    return np.exp(-x**2) # [0, 1]

# Absolute activation function
def abs(x):
    return np.abs(-x**2) # |(-x)^2|

# Square activation function
def square(x):
    return x**2 # x^2

# Step activation function
def step(x):
    return np.where(x > 0, 1, 0) # [0, 1]

# Dictionary to map activation function names to functions
activation_functions = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'identity': identity,
    'sin': sin,
    'relu': relu,
    'cos': cos,
    'gaussian': gaussian,
    'square': square,
    'step': step
}

def string_to_fn(string):
    # Converts string to activation function #
    if string in activation_functions:
        return activation_functions[string]
    else:
        raise ValueError('Unknown activation function: {}'.format(string))
    
def fn_to_string(fn):
    # Converts activation function to string #
    for name, func in activation_functions.items():
        if func == fn:
            return name
    raise ValueError('Unknown activation function: {}'.format(fn))