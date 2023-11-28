# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:59:45 2023

@author: flavi
"""
#pip install skorch

import pandas as pd
import numpy as np
import torch.nn as nn
import skorch    
from skorch import NeuralNetBinaryClassifier
import torch
import sklearn
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

#%% 
torch.__version__, skorch.__version__, sklearn.__version__

#%% 
## Base de Dados

np.random.seed(123)
torch.manual_seed(123)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


#%% 

previsores = np.array(previsores, dtype='float32')

# Transformando um DataFrame em um arrary unidimensional

classe = np.array(classe, dtype='float32').squeeze(1) 


#%% 
## Classe para estrutura da rede neural

class classificador_torch(nn.Module):
  def __init__(self):
    super().__init__()

    # 30 -> 16 -> 16 -> 1
    self.dense0 = nn.Linear(30, 16)
    torch.nn.init.uniform_(self.dense0.weight)
    self.activation0 = nn.ReLU()
    self.dense1 = nn.Linear(16, 16)
    torch.nn.init.uniform_(self.dense1.weight)
    self.activation1 = nn.ReLU()
    self.dense2 = nn.Linear(16, 1)
    torch.nn.init.uniform_(self.dense2.weight)
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
                                                  criterion=torch.nn.BCEWithLogitsLoss, # ** ATUALIZAÇÃO **
                                                  optimizer=torch.optim.Adam,
                                                  lr=0.001,
                                                  optimizer__weight_decay=0.0001,
                                                  max_epochs=100,
                                                  batch_size=10,
                                                  train_split=False)

#%% 

resultados = cross_val_score(classificador_sklearn, previsores, classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
media

#%% 


#%% 


#%% 



#%% 



#%% 



#%% 



#%% 

