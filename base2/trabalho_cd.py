import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def pre_processamento():
    print("Iniciando o pré-processamento dos dados...\n")

    script_dir = Path(__file__).resolve().parent

    # 1. Carregar os dados originais
    # O arquivo 'telco_churn.csv' deve estar na mesma pasta deste script
    df = pd.read_csv(script_dir / 'telco_churn.csv')
    print(f"Formato original: {df.shape}")

    # 2. Remover a coluna 'customerID' (Irrelevante para o modelo)
    df.drop('customerID', axis=1, inplace=True)

    # 3. Tratamento de Valores Ausentes na coluna 'TotalCharges'
    # Converte para numérico, forçando espaços em branco a virarem NaN (Nulos)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Removemos as poucas linhas que ficaram nulas (apenas 11 linhas de clientes com 0 meses)
    df.dropna(subset=['TotalCharges'], inplace=True)

    # 4. Codificação de Variáveis Categóricas (Encoding)
    # 4.1. Mapear a variável alvo (Churn) para 0 e 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 4.2. Mapear variáveis binárias simples (gender, Partner, etc)
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    bin_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in bin_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # 4.3. Aplicar One-Hot Encoding nas colunas com múltiplas categorias (ex: InternetService, Contract)
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']
    
    # get_dummies transforma cada categoria em uma nova coluna de True/False
    # drop_first=True evita a armadilha da multicolinearidade (redundância)
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # 5. Escalonamento dos Dados (Scaling) - Essencial para KNN e SVM
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Transforma os valores numéricos para terem média 0 e variância 1
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 6. Salvar o dataset final
    df.to_csv(script_dir / 'telco_churn_preprocessed.csv', index=False)
    
    print("\nPré-processamento concluído com sucesso!")
    print(f"Novo formato do dataset (linhas, colunas): {df.shape}")
    print("O arquivo 'telco_churn_preprocessed.csv' foi salvo na sua pasta.")

if __name__ == "__main__":
    pre_processamento()