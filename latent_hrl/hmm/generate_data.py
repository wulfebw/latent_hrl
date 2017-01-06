
import numpy as np

def generate_poisson_data():
    return np.tile([10,10,10,10,20,20,20,20], 20)

def generate_multinomial_data():
    return np.tile([0,1,2,1,0,1,2,1,0,2,1], 20)