# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:10:06 2023

@author: flavi
"""


#%% 
# Importando as bibliotecas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

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

tdy = train_data['Survived']
tdy = tdy.to_frame() # Convertendo o Series em Dataframe
tdX = train_data.drop('Survived', axis = 1)


#%%

# Separando as observações para treino e teste

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(tdX, tdy, test_size=0.25, stratify = tdy)
                                                                                              

                                                                                           
#%% 

# Transformando os valores True e False em 0 e 1 (Previsoes Treinamento)

substituir_sim_nao(previsores_treinamento, 'Pclass_1', True, False)
substituir_sim_nao(previsores_treinamento, 'Pclass_2', True, False)
substituir_sim_nao(previsores_treinamento, 'Pclass_3', True, False)
substituir_sim_nao(previsores_treinamento, 'Sex_female', True, False)
substituir_sim_nao(previsores_treinamento, 'Sex_male', True, False)
substituir_sim_nao(previsores_treinamento, 'Embarked_C', True, False)
substituir_sim_nao(previsores_treinamento, 'Embarked_Q', True, False)
substituir_sim_nao(previsores_treinamento, 'Embarked_S', True, False)

                                                                                     
#%% 

substituir_sim_nao(previsores_teste, 'Pclass_1', True, False)
substituir_sim_nao(previsores_teste, 'Pclass_2', True, False)
substituir_sim_nao(previsores_teste, 'Pclass_3', True, False)
substituir_sim_nao(previsores_teste, 'Sex_female', True, False)
substituir_sim_nao(previsores_teste, 'Sex_male', True, False)
substituir_sim_nao(previsores_teste, 'Embarked_C', True, False)
substituir_sim_nao(previsores_teste, 'Embarked_Q', True, False)
substituir_sim_nao(previsores_teste, 'Embarked_S', True, False)


                                                                                     
#%% 

previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype = torch.float).cuda()
classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype = torch.float).cuda()
                                                                                           
#%% 

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)

                                                                                           
#%% 

train_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)


                                                                                           
#%% 

# 12 neuronios (previsores_treinamento.shape) ---> camada de entrada
# 7 neuronios ---> 1 camada oculta ---> (entrada + saida)/2 = (12 + 1) / 2 = 7
# 7 neuronios ---> 2 camada oculta
# 1 neuronio ---> 1 camada de saida

classificador = nn.Sequential(
    nn.Linear(in_features=12, out_features=7), nn.Tanh(),
    nn.Linear(in_features=7, out_features=7), nn.Tanh(),
    nn.Linear(in_features=7, out_features=1), nn.Sigmoid()
)

# Move o modelo para a GPU
classificador = classificador.cuda()

                                                                                           
#%% 

classificador.parameters

                                                                                           
#%% 

criterion = nn.BCEWithLogitsLoss()


                                                                                           
#%% 

optimizer = torch.optim.Adam(classificador.parameters(), lr=0.001, weight_decay=0.0001)


                                                                                           
#%% 

## Treinamento do modelo


for epoch in range(100):
  running_loss = 0.

  for data in train_loader:
    inputs, labels = data
    
    optimizer.zero_grad()

    outputs = classificador(inputs) # classificador.forward(inputs)
    
    loss = criterion(outputs, labels)
    
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


previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float).cuda()


#%% 

previsoes = classificador.forward(previsores_teste)


#%% 

# Move as previsões de volta para a CPU
previsoes = previsoes.cpu().detach()

# Converte para um array NumPy
previsoes = previsoes.numpy()




#%% 

previsoes = np.array(previsoes > 0.5)
previsoes


#%% 

taxa_acerto = accuracy_score(classe_teste, previsoes)
print(taxa_acerto)


#%% 

# Preparando o Arquivo de Resposta

resultado = test_data['PassengerId']
resultado = pd.DataFrame(resultado, columns=['PassengerId'])
test_data = test_data.drop('PassengerId', axis = 1)



# Preparado o arquivo test.csv

substituir_sim_nao(test_data, 'Pclass_1', True, False)
substituir_sim_nao(test_data, 'Pclass_2', True, False)
substituir_sim_nao(test_data, 'Pclass_3', True, False)
substituir_sim_nao(test_data, 'Sex_female', True, False)
substituir_sim_nao(test_data, 'Sex_male', True, False)
substituir_sim_nao(test_data, 'Embarked_C', True, False)
substituir_sim_nao(test_data, 'Embarked_Q', True, False)
substituir_sim_nao(test_data, 'Embarked_S', True, False)



#%% 

arquivo_teste = torch.tensor(np.array(test_data), dtype=torch.float).cuda()



#%% 

previsoes_arquivo_teste = classificador.forward(arquivo_teste)

# Move as previsões de volta para a CPU e realiza detach
previsoes_arquivo_teste = previsoes_arquivo_teste.cpu().detach()

# Converte para um array NumPy
previsoes_arquivo_teste = previsoes_arquivo_teste.numpy()

#%% 

previsoes_arquivo_teste = np.array(previsoes_arquivo_teste > 0.5)
previsoes_arquivo_teste


#%% 

# Convertendo um Array bidimensional em 1 dimensao

previsoes_arquivo_teste = previsoes_arquivo_teste.flatten().astype(int) 


#%% 

# Gerando o arquivo de resposta Rede Neural 

resultado_rede_neural = pd.DataFrame({'PassengerId': resultado.PassengerId, 'Survived': previsoes_arquivo_teste})
resultado_rede_neural.to_csv('resultado_rede_neural.csv', index=False)



#%% 



#%% 



#%% 




#%% 
