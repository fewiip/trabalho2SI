
# Importar as bibliotecas necessárias
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Carregar o dataset Iris
dados = load_iris()
X = dados.data
y = dados.target

# Dividir os dados em conjunto de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_teste)

# Calcular a precisão do modelo
precisao = accuracy_score(y_teste, y_pred)
print(f'A precisão do modelo é: {precisao:.2f}')
