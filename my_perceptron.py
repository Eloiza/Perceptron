from app_perceptron_lib.perceptron import Perceptron
from app_perceptron_lib.dataset_generator import DatasetGenerator

def run_perceptron(weights, data, labels, learning_rate=1):
    epoch_error = 0
    # Para cada instancia e label
    for x, y in zip(data, labels):
        #calculando label para a instancia atual
        sample_output = 0 
        for i in range(len(x)):
            sample_output += weights[i]*x[i]

        #caso valor final positivo label se torna 1
        if(sample_output > 0):
            sample_output = 1

        #contrario label se torna 2
        else:
            sample_output = 0
    
        #calculando erro da amostra
        error = y - sample_output

        #se houver erro - atualiza pesos da rede
        if(error > 0):
            epoch_error += 1
            for i in range(len(weights)):
                weights[i] = weights[i] + (learning_rate*error*x[i])

    return weights, epoch_error
