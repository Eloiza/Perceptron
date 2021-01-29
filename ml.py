from sklearn.neural_network import MLPClassifier

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

if __name__ == "__main__":
    DATASET_LIST = [
        "xor",
        'mnist_even'
    ]

    for DATASET_NAME in DATASET_LIST:
        print("\n\n----\n{}".format(DATASET_NAME))

        if DATASET_NAME == "xor":
            trainX, testX, trainY, testY = load_binary_xor_dataset(cluster_size=500, random_state=42)

        if DATASET_NAME == "mnist_even":
            dataX = np.genfromtxt("mnist_even.txt",delimiter=" ",skip_header=0)
            dataY = np.genfromtxt("mnist_even_labels.txt",delimiter=" ",skip_header=0)

            trainX, testX, trainY, testY = train_test_split(
                dataX, dataY, test_size=0.25, random_state=42)

            img_ev = np.where(dataY == 1)[0][0]
            img_od = np.where(dataY == 0)[0][0]
            img_shape = (int(np.sqrt(dataX.shape[1])),int(np.sqrt(dataX.shape[1])))

            # Plota 2 imagens de digitos par e impar da base MNIST
            # Voce pode comentar as proximas linhas para poupar tempo de execucao
            # (ateh plt.show())
            _, axes = plt.subplots(1, 2)
            axes = np.array(axes).reshape(-1)
            axes[0].set_axis_off()
            axes[0].imshow(dataX[img_ev].reshape(img_shape), cmap=plt.cm.gray_r, interpolation='nearest')
            axes[0].set_title('Exemplo de dígito par')
            axes[1].set_axis_off()
            axes[1].imshow(dataX[img_od].reshape(img_shape), cmap=plt.cm.gray_r, interpolation='nearest')
            axes[1].set_title('Exemplo de dígito ímpar')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=-0.1)
            plt.show()

        print("Train size: {}".format(len(trainX)))
        print("Test size: {}".format(len(testX)))
        print("Features dimension: {}".format(trainX.shape[1]))

        plt_title = "Training and test set.\n '.' represents training instances and '*' test instances"

        # Cria e plota uma representacao 2D do dataset
        # Voce pode comentar essa linha para poupar tempo de execucao
        plot_binary_2d_dataset(trainX, testX, trainY, testY, title=plt_title)

        print("Treinando base de dados: ", DATASET_NAME)

        print("Treinando Modelos...")
        perceptron = PerceptronClassifier().fit(trainX, trainY)
        mlp = MLPClassifier().fit(trainX, trainY)

        print("Testando Modelos...")
        mlp_pred   = mlp.predict(testX)
        percp_pred = perceptron.predict(testX)

        mlp_acc   = accuracy_score(testY, mlp_pred)
        percp_acc = accuracy_score(testY, percp_pred) 
        print("Acurácia mlp: %.2f\nAcurácia Perceptron: %.2f\n" %(mlp_acc, percp_acc))

        mlp_f1    = f1_score(testY, mlp_pred)
        percpt_f1 = f1_score(testY, percp_pred)
        print("F1 Score mlp: %.2f\nF1 Score Perceptron: %.2f\n" %(mlp_f1, percpt_f1))

        mlp_matrix    = confusion_matrix(testY, mlp_pred)
        percpt_matrix = confusion_matrix(testY, percp_pred)
        print("Confusion Matrix MLP", mlp_matrix)
        print("Confusion Matrix Perceptron", percpt_matrix)


