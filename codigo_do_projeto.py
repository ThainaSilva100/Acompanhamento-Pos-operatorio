import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Gerar um dataset fictício
np.random.seed(42)  # para garantir a reprodutibilidade

data_size = 500
df = pd.DataFrame({
    'AGE': np.random.randint(20, 80, data_size),
    'GENDER': np.random.choice(['F', 'M'], data_size),
    'POST_OPERATIVE_STATUS': np.random.choice(['Y', 'N'], data_size),
    'DIAGNOSIS': np.random.choice(['D1', 'D2', 'D3', 'D4'], data_size),
    'DIGNITY_SCORE': np.random.randint(1, 6, data_size),
    'CONFIDENTIALITY_SCORE': np.random.randint(1, 6, data_size),
    'ROUND_DURATION_MIN': np.random.randint(10, 120, data_size),
    'SATISFACTION_SCORE': np.random.randint(1, 6, data_size)
})

# Carregando os dados
# df = pd.read_csv('path_to_data.csv')

# Informações gerais
print(df.info())

# Primeiras linhas do dataset
print(df.head())

# Resumo estatístico
print(df.describe())

# Contagem de valores faltantes
missing_values = df.isnull().sum()
print(missing_values)

import matplotlib.pyplot as plt
import seaborn as sns

# Distribuição das idades, como exemplo
plt.figure(figsize=(10, 6))
sns.histplot(df['AGE'], bins=30, kde=True)
plt.title('Distribuição das Idades')
plt.show()

# Matriz de correlação
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Codificação One-Hot
df_encoded = pd.get_dummies(df, columns=['GENDER', 'POST_OPERATIVE_STATUS', 'DIAGNOSIS'])

# Dividindo os dados em treino e teste
X = df_encoded.drop('SATISFACTION_SCORE', axis=1)
y = df_encoded['SATISFACTION_SCORE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

# Inicialização do modelo
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinamento do modelo
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error

# Cálculo do MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse}")

feature_names = X.columns

importances = rf.feature_importances_
importance_data = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
print(importance_data)

import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.scatter(y_test, residuals, alpha=0.5)
plt.title("Plot de Residuais")
plt.xlabel("Valores Verdadeiros")
plt.ylabel("Residuais")
plt.show()
