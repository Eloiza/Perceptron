from app_perceptron_lib.perceptron import Perceptron
from app_perceptron_lib.dataset_generator import DatasetGenerator

import numpy as np

def activation_function(entry):
    if(entry >= 0):
        return 1
    else:  
        return 0

def run_perceptron(weights, data, labels, learning_rate=1):
    epoch_error = 0
    # Para cada instancia e label
    for x, y in zip(data, labels):

        activation = np.dot(x,weights)

        prediction = activation_function(activation)

        #calculando erro da amostra
        error = y - prediction
    
        #se houver erro - atualiza pesos da rede
        if(error > 0):
            epoch_error += 1
            for i in range(len(weights)):
                weights[i] = weights[i] + (learning_rate*error*x[i])

    return weights, epoch_error
