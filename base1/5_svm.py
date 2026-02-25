import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# 1. Carregar e Dividir
df = pd.read_csv('data/heart_disease_processado.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- RESULTADOS SVM (Heart Disease) ---")

# 2. Parametrização 1: Kernel Linear
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print(f"SVM (Linear) -> Acurácia: {accuracy_score(y_test, y_pred_linear):.4f} | F1-Score: {f1_score(y_test, y_pred_linear):.4f}")

# 3. Parametrização 2: Kernel RBF
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(f"SVM (RBF) -> Acurácia: {accuracy_score(y_test, y_pred_rbf):.4f} | F1-Score: {f1_score(y_test, y_pred_rbf):.4f}")

# 4. Salvar Matriz
ConfusionMatrixDisplay.from_estimator(
	svm_rbf,
	X_test,
	y_test,
	display_labels=['Saudável', 'Doente'],
	cmap='Purples'
)
plt.title("Matriz de Confusão - SVM (Kernel RBF)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig('plots/base_01/6_matriz_svm.png')
plt.close()
print("Matriz de confusão do SVM salva em plots/base_01/")