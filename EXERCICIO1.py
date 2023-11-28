# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:26:20 2023

@author: flavi
"""

# Importação das bibliotecas

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%% 
import torch
torch.__version__
#!pip install torch==1.4.0


#%% 
import torch.nn as nn


#%% 
## Base de Dados
np.random.seed(123)
torch.manual_seed(123)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


#%% 
# Separando base de teste e base de treinamento

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,
                                                                                              classe,
                                                                                              test_size = 0.25)

#%% 
## Transformação dos dados para tensores

type(previsores_treinamento)


#%% 

type(np.array(previsores_treinamento))


#%% 

previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype = torch.float)
classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype = torch.float)


#%% 

type(previsores_treinamento)


#%% 

type(classe_treinamento)


#%% 

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)


#%% 

type(dataset)


#%% 

train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)


#%% 
##Construção do modelo

# 30 -> 16 -> 16 -> 1
# (entradas + saida) / 2 = (30 + 1) / 2 = 16
classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)


#%% 

classificador.parameters


#%% 

criterion = nn.BCELoss()


#%% 

optimizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.0001)


#%% 
## Treinamento do modelo


for epoch in range(100):
  running_loss = 0.

  for data in train_loader:
    inputs, labels = data
    #print(inputs)
    #print('-----')
    #print(labels)
    optimizer.zero_grad()

    outputs = classificador(inputs) # classificador.forward(inputs)
    #print(outputs)
    loss = criterion(outputs, labels)
    #print(loss)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))



#%% 
# 30 -> 16 -> 16 -> 1
params = list(classificador.parameters())

#%% 

params

#%% 
## Avaliação do modelo

classificador.eval()


#%% 

previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)


#%% 

previsoes = classificador.forward(previsores_teste)


#%% 

previsoes


#%% 

previsoes = np.array(previsoes > 0.5)
previsoes


#%% 

taxa_acerto = accuracy_score(classe_teste, previsoes)
taxa_acerto

#%% 



#%% 



#%% 



