import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# 1. Carregar e Dividir
df = pd.read_csv('data/heart_disease_processado.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- RESULTADOS ÁRVORE DE DECISÃO (Heart Disease) ---")

# 2. Parametrização 1: Profundidade Livre (None)
tree_livre = DecisionTreeClassifier(random_state=42)
tree_livre.fit(X_train, y_train)
y_pred_livre = tree_livre.predict(X_test)
print(f"Árvore (Livre) -> Acurácia: {accuracy_score(y_test, y_pred_livre):.4f} | F1-Score: {f1_score(y_test, y_pred_livre):.4f}")

# 3. Parametrização 2: Profundidade Máxima = 3
tree_podada = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_podada.fit(X_train, y_train)
y_pred_podada = tree_podada.predict(X_test)
print(f"Árvore (Profundidade=3) -> Acurácia: {accuracy_score(y_test, y_pred_podada):.4f} | F1-Score: {f1_score(y_test, y_pred_podada):.4f}")

# 4. Salvar Matriz
ConfusionMatrixDisplay.from_estimator(
	tree_podada,
	X_test,
	y_test,
	display_labels=['Saudável', 'Doente'],
	cmap='Greens'
)
plt.title("Matriz de Confusão - Árvore (Profundidade=3)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig('plots/base_01/5_matriz_arvore.png')
plt.close()
print("Matriz de confusão da Árvore salva em plots/base_01/")