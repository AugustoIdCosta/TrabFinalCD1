from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    data_dir = script_dir / "data"
    plots_dir = script_dir / "plots" / "base_02"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "telco_churn.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {raw_path}. "
            "Coloque o 'telco_churn.csv' em base2/data/ antes de rodar."
        )

    print("--- 1. CARREGANDO OS DADOS (BASE 02) ---")
    df_raw = pd.read_csv(raw_path)
    print(f"Formato da base: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")
    print("\nTipos de dados e valores ausentes (visão rápida):")
    print(df_raw.info())
    print("\nValores ausentes por coluna:")
    print(df_raw.isnull().sum())

    # Gráfico 1: Distribuição da classe (original)
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(x="Churn", data=df_raw, hue="Churn", palette="viridis", legend=False)
    ax.set_title("Distribuição Original do Churn")
    ax.set_xlabel("Churn (Yes/No)")
    ax.set_ylabel("Quantidade de clientes")
    plt.tight_layout()
    plt.savefig(plots_dir / "1_distribuicao_classes_original.png")
    plt.close()

    print("\n--- 2. PRÉ-PROCESSAMENTO ---")
    df = df_raw.copy()

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # TotalCharges vem como texto e pode ter espaços vazios
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(subset=["TotalCharges"], inplace=True)

    # Target binário
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Gráfico 2: Distribuição após binarização
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(x="Churn", data=df, hue="Churn", palette="Set2", legend=False)
    ax.set_title("Distribuição do Churn (Binarizado)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Quantidade de clientes")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Não (0)", "Sim (1)"])
    plt.tight_layout()
    plt.savefig(plots_dir / "2_distribuicao_classes_binario.png")
    plt.close()

    # Variáveis binárias simples
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    bin_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in bin_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot para colunas com múltiplas categorias
    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    multi_cols = [col for col in multi_cols if col in df.columns]
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Escalonamento
    num_cols = [col for col in ["tenure", "MonthlyCharges", "TotalCharges"] if col in df.columns]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Gráfico 3: Matriz de correlação
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    heatmap.set_title("Matriz de Correlação das Variáveis (Base 02)", pad=16)
    plt.tight_layout()
    plt.savefig(plots_dir / "3_matriz_correlacao.png")
    plt.close()

    out_path = data_dir / "telco_churn_preprocessed.csv"
    df.to_csv(out_path, index=False)

    print("\nPré-processamento concluído com sucesso!")
    print(f"Novo formato do dataset (linhas, colunas): {df.shape}")
    print(f"Arquivo pré-processado salvo em: {out_path}")
    print(f"Relatórios salvos em: {plots_dir}")


if __name__ == "__main__":
    main()
