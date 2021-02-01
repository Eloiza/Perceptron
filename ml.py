import matplotlib.pyplot as plt  
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    font = {'family' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Classe Original')
    plt.xlabel('Classe Predita\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

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

            img_ev = np.where(dataY == 1)[0][0]
            img_od = np.where(dataY == 0)[0][0]
            img_shape = (int(np.sqrt(dataX.shape[1])),int(np.sqrt(dataX.shape[1])))

            # Plota 2 imagens de digitos par e impar da base MNIST
            # Voce pode comentar as proximas linhas para poupar tempo de execucao
            # (ateh plt.show())
            # _, axes = plt.subplots(1, 2)
            # axes = np.array(axes).reshape(-1)
            # axes[0].set_axis_off()
            # axes[0].imshow(dataX[img_ev].reshape(img_shape), cmap=plt.cm.gray_r, interpolation='nearest')
            # axes[0].set_title('Exemplo de dígito par')
            # axes[1].set_axis_off()
            # axes[1].imshow(dataX[img_od].reshape(img_shape), cmap=plt.cm.gray_r, interpolation='nearest')
            # axes[1].set_title('Exemplo de dígito ímpar')
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=-0.1)
            # plt.show()

        print("Train size: {}".format(len(trainX)))
        print("Test size: {}".format(len(testX)))
        print("Features dimension: {}".format(trainX.shape[1]))

        # plt_title = "Training and test set.\n '.' represents training instances and '*' test instances"

        # Cria e plota uma representacao 2D do dataset
        # Voce pode comentar essa linha para poupar tempo de execucao
        # plot_binary_2d_dataset(trainX, testX, trainY, testY, title=plt_title)

        print("Treinando Modelos...")
        #meu perceptron 
        perceptron = PerceptronClassifier(learning_rate=0.001, verbose=1, random_state=13, max_iter=300)
        perceptron.fit(trainX, trainY)

        mlp = MLPClassifier(random_state=1, max_iter=100).fit(trainX, trainY)

        #perceptron do scikit-learn
        perceptron_sk = Perceptron(random_state=13, max_iter=300).fit(trainX, trainY)

        print("Testando Modelos...")
        mlp_pred   = mlp.predict(testX)
        percp_pred = perceptron.predict(testX)
        percp_pred_sk = perceptron_sk.predict(testX)

        mlp_acc   = accuracy_score(testY, mlp_pred)
        percp_acc = accuracy_score(testY, percp_pred) 
        percp_acc_sk = accuracy_score(testY, percp_pred_sk) 

        mlp_f1    = f1_score(testY, mlp_pred)
        percpt_f1 = f1_score(testY, percp_pred)
        percpt_f1_sk = f1_score(testY, percp_pred_sk)

        mlp_matrix    = confusion_matrix(testY, mlp_pred)
        percpt_matrix = confusion_matrix(testY, percp_pred)
        percpt_matrix_sk = confusion_matrix(testY, percp_pred_sk)


        print("==========Perceptron==========")
        print("Acurácia: %.2f" %(percp_acc*100))
        print("F1 Score: %.2f" %(percpt_f1*100))
        print("------Matriz de Confusão------")
        print_mc(percpt_matrix)

        print("=========Perceptron SK=========")
        print("Acurácia: %.2f" %(percp_acc_sk*100))
        print("F1 Score: %.2f" %(percpt_f1_sk*100))
        print("------Matriz de Confusão------")
        print_mc(percpt_matrix_sk)


        print("==============MLP=============")
        print("Acurácia: %.2f" %(mlp_acc*100))
        print("F1 Score: %.2f" %(mlp_f1*100))
        print("------Matriz de Confusão------")
        print_mc(mlp_matrix)


        if(DATASET_NAME == 'xor'):
            plot_confusion_matrix(cm= mlp_matrix, target_names= ["0", "1"], title='Matriz de Confusão MLP - Base XOR', cmap=None, normalize=False)
            plot_confusion_matrix(cm= percpt_matrix, target_names= ["0", "1"], title='Matriz de Confusão Perceptron - Base XOR', cmap=None, normalize=False)

        else:
            plot_confusion_matrix(cm= mlp_matrix, target_names= ["Ímpar", "par"], title='Matriz de Confusão MLP - Base MNIST', cmap=None, normalize=False)
            plot_confusion_matrix(cm= percpt_matrix, target_names= ["Ímpar", "par"], title='Matriz de Confusão Perceptron - Base MNIST', cmap=None, normalize=False)
