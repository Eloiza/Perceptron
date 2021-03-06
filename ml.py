from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix

#imports para os classificadores
from custom_classifiers import PerceptronClassifier
from custom_classifiers import plot_binary_2d_dataset
from custom_classifiers import load_binary_iris_dataset
from custom_classifiers import load_binary_random_dataset
from custom_classifiers import load_binary_xor_dataset

#manipulacao de dados
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#imports para metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def print_mc(matrix):
	for l in matrix:
		print(str(l)[1:-1]) 


if __name__ == "__main__":
    DATASET_LIST = [
        "xor",
        'mnist_even'
    ]

    for DATASET_NAME in DATASET_LIST:
        print("\n\n---- {}".format(DATASET_NAME))

        if DATASET_NAME == "xor":
            trainX, testX, trainY, testY = load_binary_xor_dataset(cluster_size=500, random_state=42)

        if DATASET_NAME == "mnist_even":
            dataX = np.genfromtxt("mnist_even.txt",delimiter=" ",skip_header=0)
            dataY = np.genfromtxt("mnist_even_labels.txt",delimiter=" ",skip_header=0)

            trainX, testX, trainY, testY = train_test_split(
                dataX, dataY, test_size=0.25, random_state=42)

        print("Train size: {}".format(len(trainX)))
        print("Test size: {}".format(len(testX)))
        print("Features dimension: {}".format(trainX.shape[1]))

        print("Treinando Modelos...")
        perceptron = PerceptronClassifier(learning_rate=0.001, verbose=1, random_state=13, max_iter=300)
        perceptron.fit(trainX, trainY)

        mlp = MLPClassifier(random_state=1, max_iter=100).fit(trainX, trainY)

        print("Testando Modelos...")
        mlp_pred   = mlp.predict(testX)
        percp_pred = perceptron.predict(testX)

        mlp_acc   = accuracy_score(testY, mlp_pred)
        percp_acc = accuracy_score(testY, percp_pred) 

        mlp_f1    = f1_score(testY, mlp_pred)
        percpt_f1 = f1_score(testY, percp_pred)

        mlp_matrix    = confusion_matrix(testY, mlp_pred)
        percpt_matrix = confusion_matrix(testY, percp_pred)


        print("==========Perceptron==========")
        print("Acurácia: %.2f" %(percp_acc*100))
        print("F1 Score: %.2f" %(percpt_f1*100))
        print("------Matriz de Confusão------")
        print_mc(percpt_matrix)

        print("==============MLP=============")
        print("Acurácia: %.2f" %(mlp_acc*100))
        print("F1 Score: %.2f" %(mlp_f1*100))
        print("------Matriz de Confusão------")
        print_mc(mlp_matrix)
