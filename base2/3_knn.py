import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots" / "base_02"
plots_dir.mkdir(parents=True, exist_ok=True)

preprocessed_path = script_dir / "telco_churn_preprocessed.csv"
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

print("--- RESULTADOS KNN (Telco Churn) ---")

# 2. Parametrização 1: K = 3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
y_pred_3 = knn_3.predict(X_test)
print(
    f"KNN (K=3) -> Acurácia: {accuracy_score(y_test, y_pred_3):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_3):.4f}"
)

# 3. Parametrização 2: K = 7
knn_7 = KNeighborsClassifier(n_neighbors=7)
knn_7.fit(X_train, y_train)
y_pred_7 = knn_7.predict(X_test)
print(
    f"KNN (K=7) -> Acurácia: {accuracy_score(y_test, y_pred_7):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_7):.4f}"
)

# 4. Salvar Matriz de Confusão do modelo K=7
ConfusionMatrixDisplay.from_estimator(
    knn_7,
    X_test,
    y_test,
    display_labels=["Não", "Sim"],
    cmap="Blues",
)
plt.title("Matriz de Confusão - KNN (K=7)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig(plots_dir / "4_matriz_knn.png")
plt.close()
print(f"Matriz de confusão do KNN salva em {plots_dir}")
