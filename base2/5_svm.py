import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

print("--- RESULTADOS SVM (Telco Churn) ---")

# 2. Parametrização 1: Kernel Linear
svm_linear = SVC(kernel="linear", random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print(
    f"SVM (Linear) -> Acurácia: {accuracy_score(y_test, y_pred_linear):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_linear):.4f}"
)

# 3. Parametrização 2: Kernel RBF
svm_rbf = SVC(kernel="rbf", random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(
    f"SVM (RBF) -> Acurácia: {accuracy_score(y_test, y_pred_rbf):.4f} | "
    f"F1-Score: {f1_score(y_test, y_pred_rbf):.4f}"
)

# 4. Salvar Matriz de Confusão do modelo RBF
ConfusionMatrixDisplay.from_estimator(
    svm_rbf,
    X_test,
    y_test,
    display_labels=["Não", "Sim"],
    cmap="Purples",
)
plt.title("Matriz de Confusão - SVM (Kernel RBF)")
plt.xlabel("Classe prevista")
plt.ylabel("Classe verdadeira")
plt.savefig(plots_dir / "6_matriz_svm.png")
plt.close()
print(f"Matriz de confusão do SVM salva em {plots_dir}")
