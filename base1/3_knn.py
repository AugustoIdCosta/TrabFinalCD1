import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import os

os.makedirs('plots/base_01', exist_ok=True)

# 1. Carregar e Dividir
df = pd.read_csv('data/heart_disease_processado.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- RESULTADOS KNN (Heart Disease) ---")

# 2. Parametrização 1: K = 3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
y_pred_3 = knn_3.predict(X_test)
print(f"KNN (K=3) -> Acurácia: {accuracy_score(y_test, y_pred_3):.4f} | F1-Score: {f1_score(y_test, y_pred_3):.4f}")

# 3. Parametrização 2: K = 7
knn_7 = KNeighborsClassifier(n_neighbors=7)
knn_7.fit(X_train, y_train)
y_pred_7 = knn_7.predict(X_test)
print(f"KNN (K=7) -> Acurácia: {accuracy_score(y_test, y_pred_7):.4f} | F1-Score: {f1_score(y_test, y_pred_7):.4f}")

# 4. Salvar Matriz de Confusão do melhor modelo (assumindo K=7 como padrão de teste)
ConfusionMatrixDisplay.from_estimator(
	knn_7,
	X_test,
	y_test,
	display_labels=['Saudável', 'Doente'],
	cmap='Blues'
)
plt.title("Matriz de Confusão - KNN (K=7)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig('plots/base_01/4_matriz_knn.png')
plt.close()
print("Matriz de confusão do KNN salva em plots/base_01/")