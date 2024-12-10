
# Importar bibliotecas necessárias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Carregar o dataset Boston Housing
#boston = load_boston()
#X = boston.data
#y = boston.target

#Como importar
# https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X= np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliar o modelo usando Mean Squared Error (Erro Quadrático Médio)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse:.2f}')

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valores Reais')
plt.plot(y_pred, label='Valores Preditos', linestyle='--')
plt.legend()
plt.xlabel('Amostras')
plt.ylabel('Mediana dos Valores das Casas')
plt.title('KNeighborsRegressor - Comparação de Valores Reais e Preditos')
plt.show()
