import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots" / "base_02"
plots_dir.mkdir(parents=True, exist_ok=True)

preprocessed_path = script_dir / "data" / "telco_churn_preprocessed.csv"
if not preprocessed_path.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {preprocessed_path}. "
        "Rode primeiro o pré-processamento (trabalho_cd.py) para gerá-lo."
    )

# 1. Carregar e Dividir
_df = pd.read_csv(preprocessed_path)
X = _df.drop("Churn", axis=1)
y = _df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("--- RESULTADOS ÁRVORE DE DECISÃO (Telco Churn) ---")

# 2. Parametrização 1: Profundidade Livre (None)
tree_livre = DecisionTreeClassifier(random_state=42)
tree_livre.fit(X_train, y_train)
y_pred_livre = tree_livre.predict(X_test)
print(
    f"Árvore (Livre) -> Acurácia: {accuracy_score(y_test, y_pred_livre):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_livre):.4f}"
)

# 3. Parametrização 2: Profundidade Máxima = 3
tree_podada = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_podada.fit(X_train, y_train)
y_pred_podada = tree_podada.predict(X_test)
print(
    f"Árvore (Profundidade=3) -> Acurácia: {accuracy_score(y_test, y_pred_podada):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_podada):.4f}"
)

# 4. Salvar Matriz de Confusão do modelo podado
ConfusionMatrixDisplay.from_estimator(
    tree_podada,
    X_test,
    y_test,
    display_labels=["Não", "Sim"],
    cmap="Greens",
)
plt.title("Matriz de Confusão - Árvore (Profundidade=3)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig(plots_dir / "5_matriz_arvore.png")
plt.close()
print(f"Matriz de confusão da Árvore salva em {plots_dir}")
