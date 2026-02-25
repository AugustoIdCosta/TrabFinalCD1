import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Garante que as pastas necessárias existem
os.makedirs('plots/base_01', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. CARREGANDO OS DADOS
print("--- 1. CARREGANDO OS DADOS ---")
df = pd.read_csv('heart_disease.csv')

# 2. EXPLORAÇÃO DOS DADOS (Requisito 'b' do trabalho)
print("\n--- 2. EXPLORAÇÃO DOS DADOS ---")
print(f"Formato da base: {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("\nTipos de dados e valores ausentes originais:")
print(df.info())

# O dataset original usa '?' para valores nulos nas colunas 'ca' e 'thal'.
# Vamos substituir isso pelo padrão do Pandas (NaN) para conseguirmos contar.
df.replace('?', np.nan, inplace=True)
print("\nValores ausentes (após ajuste de '?'):")
print(df.isnull().sum())

# Gráfico 1: Distribuição da Classe (antes da binarização)
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='target', data=df, palette='viridis')
ax.set_title('Distribuição Original do Diagnóstico')
ax.set_xlabel('Classe do diagnóstico (0 = saudável, 1–4 = graus de doença)')
ax.set_ylabel('Quantidade de pacientes')
plt.tight_layout()
plt.savefig('plots/base_01/1_distribuicao_classes_original.png')
plt.close()

# 3. PRÉ-PROCESSAMENTO (Requisito 'c' do trabalho)
print("\n--- 3. PRÉ-PROCESSAMENTO ---")

# A. Transformar em Problema Binário:
# O target original vai de 0 (sem doença) a 4. Vamos simplificar: 0 = Saudável, 1 = Doente (1, 2, 3 ou 4)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
print("Transformação do Target aplicada (0 = Saudável, 1 = Doente).")

# Gráfico 2: Distribuição após binarização
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='target', data=df, palette='Set2')
ax.set_title('Distribuição do Diagnóstico (Binarizado)')
ax.set_xlabel('Classe do diagnóstico')
ax.set_ylabel('Quantidade de pacientes')
ax.set_xticklabels(['Saudável (0)', 'Doente (1)'])
plt.tight_layout()
plt.savefig('plots/base_01/2_distribuicao_classes_binario.png')
plt.close()

# B. Tratamento de Valores Ausentes:
# Como são pouquíssimos nulos em 'ca' e 'thal', vamos preencher com a moda (valor mais frequente)
imputer = SimpleImputer(strategy='most_frequent')
df[['ca', 'thal']] = imputer.fit_transform(df[['ca', 'thal']])
# Convertendo de volta para numérico que acabou virando object
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

# C. Separação entre Atributos (X) e Classe (y)
X = df.drop('target', axis=1)
y = df['target']

# Gráfico 3: Matriz de Correlação (ajuda a entender quais atributos importam mais)
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
heatmap.set_title('Matriz de Correlação das Variáveis', pad=16)
plt.tight_layout()
plt.savefig('plots/base_01/3_matriz_correlacao.png')
plt.close()

# D. Normalização / Re-escala dos Dados:
# O KNN e o SVM são algoritmos baseados em cálculo de distância. 
# Se não normalizarmos, variáveis com valores altos (como colesterol 'chol') vão dominar o modelo.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Normalização (StandardScaler) aplicada aos atributos.")

# Salvar a base pré-processada para usarmos no treinamento depois
X_scaled['target'] = y.values
X_scaled.to_csv('data/heart_disease_processado.csv', index=False)
print("\nBase limpa e processada salva em 'data/heart_disease_processado.csv'.")