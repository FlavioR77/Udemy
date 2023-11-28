# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 00:20:46 2023

@author: flavi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:10:06 2023

@author: flavi
"""


#%% 
# Importando as bibliotecas

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#%% 

  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")        
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


#%% 


#%% 
# Lendo os arquivos propostos no desafio

np.random.seed(13)
torch.manual_seed(13)

train_data = pd.read_csv('train.csv', sep=',', header=0 )
test_data = pd.read_csv('test.csv', sep=',', header=0 )

#%% 

# Função para substituir os valores de Sim e Nao para 1 e 0

def substituir_sim_nao(df, coluna, sim, nao):
    df[coluna] = df[coluna].replace({sim: 1, nao: 0})
    df[coluna] = df[coluna].astype('uint8')
    return df


#%% 

# Função para determinar a porcentagem de nulos

def porcentagem_nulos (df):
    porcentagem_nulos = round((df.isnull().sum() / len(df)) * 100, 2)
    print(porcentagem_nulos)

#%% 

# Verificando a porcentagem de nulos no train_data

porcentagem_nulos(train_data)

#%% 

# Verificando a porcentagem de nulos no test_data

porcentagem_nulos(test_data)


#%% 

#Validando redundancia entre linhas (linhas duplicadas)

print('\nExistência de linhas duplicadas')
print(train_data.duplicated().any())


#%% 

# Excluindo linhas duplicadas 

train_data.drop_duplicates(inplace=True)

#%% 

# Excluindo as Colunas com alta porcentagem de nulos 

train_data = train_data.drop(columns=['Cabin'])
test_data = test_data.drop(columns=['Cabin'])


#%% 

# Verificando os valores únicos de cada coluna (nunique())

train_data.nunique()


#%% 

# Analisando as variáveis com pouco variância

baixa_variancia = []
serie = train_data.nunique()

# Lista variaveis com menos de 1% de unicos em relacao ao dataset

for i in range(train_data.shape[1]):
    num = serie[i]
    perc = float(num) / train_data.shape[0] * 100
    if perc < 1:
        print('%d. %s, %s, Unq: %d, Perc: %.1f%%' % (i, train_data.columns[i], str(train_data[train_data.columns[i]].dtype), num, perc))
        baixa_variancia.append(train_data.columns[i])

train_data[baixa_variancia]


#%% 

# Visualizando as colunas do train_data

train_data.columns


#%% 


# Excluindo as colunas sem relevância para a análise


train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket'])
test_data = test_data.drop(columns=['Name', 'Ticket'])


#%% 

# Substituindo os valores Nulos pela mediana

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())



#%% 

# Visualizando as observações com nulo

linha_com_nan = train_data[train_data.isnull().any(axis=1)]
print(linha_com_nan)

#%% 

# Excluindo as linhas com nulo

train_data = train_data.dropna()



#%% 

# Obtendo as informações do dataframe train_data

train_data.info()


#%% 

# Para cada uma das variáveis categoricas, aplicamos a função get_dummies e incluimos no DataFrame train_data

variaveis_categoricas = ['Pclass','Sex', 'Embarked']

for var in variaveis_categoricas: 
    lista_categoricas = pd.get_dummies(train_data[var], prefix=var)
    train_data = pd.concat([train_data, lista_categoricas], axis=1)

train_data = train_data.drop(columns=['Pclass','Sex', 'Embarked'])


#%% 


# Para cada uma das variáveis categoricas, aplicamos a função get_dummies e incluimos no DataFrame test_data

variaveis_categoricas = ['Pclass','Sex', 'Embarked']

for var in variaveis_categoricas: 
    lista_categoricas = pd.get_dummies(test_data[var], prefix=var)
    test_data = pd.concat([test_data, lista_categoricas], axis=1)

test_data = test_data.drop(columns=['Pclass','Sex', 'Embarked'])


#%% 

# Mudando o tipo de dado da variável Survived para uint8

train_data['Survived'] = train_data['Survived'].astype('uint8')


#%% 

# Vamos dividir o dataframe em dois: Variável Alvo e Demais variáveis

classe = train_data['Survived']
classe = classe.to_frame() # Convertendo o Series em Dataframe
previsores = train_data.drop('Survived', axis = 1)

                                                                                           
#%% 

# Transformando os valores True e False em 0 e 1 (Previsoes Treinamento)

substituir_sim_nao(previsores, 'Pclass_1', True, False)
substituir_sim_nao(previsores, 'Pclass_2', True, False)
substituir_sim_nao(previsores, 'Pclass_3', True, False)
substituir_sim_nao(previsores, 'Sex_female', True, False)
substituir_sim_nao(previsores, 'Sex_male', True, False)
substituir_sim_nao(previsores, 'Embarked_C', True, False)
substituir_sim_nao(previsores, 'Embarked_Q', True, False)
substituir_sim_nao(previsores, 'Embarked_S', True, False)

                                                                                     
#%% 


previsores = np.array(previsores, dtype='float32')

# Transformando um DataFrame em um arrary unidimensional

classe = np.array(classe, dtype='float32').squeeze(1) 


                                                                                           
#%% 

# 12 neuronios (previsores_treinamento.shape) ---> camada de entrada
# 7 neuronios ---> 1 camada oculta ---> (entrada + saida)/2 = (12 + 1) / 2 = 7
# 7 neuronios ---> 2 camada oculta
# 1 neuronio ---> 1 camada de saida


## Classe para estrutura da rede neural

class classificador_torch(nn.Module):
  def __init__(self, activation, neurons, initializer):
    super().__init__()
    # 12 -> 7 -> 7 -> 1
    self.dense0 = nn.Linear(12, neurons)
    initializer(self.dense0.weight)
    self.activation0 = activation
    self.dense1 = nn.Linear(neurons, neurons)
    initializer(self.dense1.weight)
    self.activation1 = activation
    self.dense2 = nn.Linear(neurons, 1)
    initializer(self.dense2.weight)
    # self.output = nn.Sigmoid() ** ATUALIZAÇÃO (ver detalhes no texto acima) **

  def forward(self, X):
    X = self.dense0(X)
    X = self.activation0(X)
    X = self.dense1(X)
    X = self.activation1(X)
    X = self.dense2(X)
    # X = self.output(X) ** ATUALIZAÇÃO (ver detalhes no texto acima) **
    return X
                                                                                           
#%% 
# Skorch

classificador_sklearn = NeuralNetBinaryClassifier(module=classificador_torch,
                                                  lr = 0.001,
                                                  optimizer__weight_decay = 0.0001,
                                                  train_split=False)

                                                                                           
#%% 
# Tuning dos parâmetros

params = {'batch_size': [5],
          'max_epochs': [150],
          'optimizer': [torch.optim.Adam],
          'criterion': [torch.nn.BCEWithLogitsLoss], # ** ATUALIZAÇÃO **
          'module__activation': [F.tanh],
          'module__neurons': [14], 
          'module__initializer': [torch.nn.init.uniform_]}



                                                                                           
#%% 

grid_search = GridSearchCV(estimator=classificador_sklearn, param_grid=params,
                           scoring = 'accuracy', cv = 5)
grid_search = grid_search.fit(previsores, classe)


                                                                                           
#%% 

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_



#%% 

melhor_precisao



#%% 

print(melhores_parametros)

#%% 



#%% 




#%% 
