# Perceptron :eye: :lips: :eye:
### Descrição
Neste trabalho para a disciplina de Inteligência Artificial foi implementado a função de treinamento do Perceptron. Esta função foi, então, avaliada e comparada com o desempenho da rede neural MLP para duas bases de dados: a base xor e a base MNIST. Ambas as bases tiveram seus dados reduzidos para o funcionamento deste trabalhó. A base MNIST além de ter amostras reduzidas, teve também o total de classes reduzidas para um problema de classificação de números ímpares ou pares. Para este trabalho implementei apenas as funções disponíveis nos arquivos `my_perceptron.py` e `ml.py`, o restante dos arquivos deste repositório foram disponibilizados pela professora da disciplina. O arquivo  `my_perceptron.py` apresenta a implementação da função de treinamento do perceptron e o arquivo `ml.py` é responsável por comparar o desempenho do Perceptron implementado com a rede MLP disponível pela biblioteca `scikit-learn`. 

### Dependências
Para executar o código é necessário ter instalado em seu ambiente Python a bibliote `scikit-learn` e a biblioteca `numpy`. Elas podem ser instaladas pelos comandos abaixo:

##### Instalação da biblioteca `numpy`
```
pip3 install -U numpy
```
##### Instalação da biblioteca scikit-learn `scikit-learn`
```
pip3 install -U scikit-learn
```

### Executando o código 
Existem duas formas de executar o código disponível neste repositório. O primeiro é utilizar o código do arquivo `app_perceptron.py` para executar um aplicativo simples que permite treinar e testar o trecho implementado em `my_perceptron.py`. O aplicativo possui uma janela, com um gráfico apresentando a fronteira de decisão calculada pelo Perceptron e a fronteira de decisão de uma base de dados criada aleatoriamente. Para tal, basta executar o comando abaixo:

```
python3 app_perceptron.py
```

Outra forma de executar o código é utilizado o arquivo `ml.py` para treinar o Perceptron implementado e uma rede MLP para a base XOR e a base MNIST. Ao final do treinamento os classificadores são avaliados por meio das métricas de acurácia, F1 Score e matriz de confusão. Para tal, basta executar o comando abaixo:

```
python3 ml.py
```
