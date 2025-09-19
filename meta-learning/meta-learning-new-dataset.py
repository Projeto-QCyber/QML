import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pymfe.mfe import MFE
import joblib
import warnings
import os
import time

warnings.filterwarnings("ignore")

# ====================================================
# CONFIGURAÇÕES
# ====================================================
RANDOM_STATE = 42
N_JOBS = -1
META_MODEL_PATH = "resultados_meta/meta_modelo_pymfe.joblib"
RESULTADOS_DIR = "resultados_meta"

# ====================================================
# FUNÇÕES
# ====================================================
def carregar_dados(caminho, alvo="Attack_label"):
    df = pd.read_csv(caminho)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=[alvo])
    y = df[alvo]

    # Amostra estratificada para extração de meta-features e treinamento
    # Usaremos a amostra para meta-features e o dataset completo para o treino final
    X_sample, _, y_sample, _ = train_test_split(
        X, y, test_size=0.8, stratify=y, random_state=RANDOM_STATE
    )
    
    X_numeric_sample = X_sample.select_dtypes(include=np.number)
    X_numeric_sample = X_numeric_sample.apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_encoded_sample = LabelEncoder().fit_transform(y_sample)

    return df, X_numeric_sample, y_encoded_sample


def extrair_meta_features(X, y):
    mfe = MFE(
        groups=["general", "statistical", "info-theory"],
        random_state=RANDOM_STATE,
        summary=("mean", "sd"),
    )
    mfe.fit(X, y)
    features, names = mfe.extract()
    return dict(zip(names, features))


def treinar_modelo_final(df_completo, params, nome_base):
    print(f"\n--- Treinando modelo final para {nome_base} com parâmetros recomendados ---")
    start = time.time()
    
    X = df_completo.drop(columns=["Attack_label"]).select_dtypes(include=np.number)
    y = LabelEncoder().fit_transform(df_completo["Attack_label"])

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    model.fit(X.values, y)

    model_path = os.path.join(RESULTADOS_DIR, f"modelo_final_recomendado_{nome_base}.joblib")
    joblib.dump(model, model_path)

    print(f"-> Modelo final treinado e salvo em: {model_path}")
    print(f"-> Treinamento concluído em {time.time() - start:.2f}s.")


def prever_e_treinar(caminho_csv):
    if not os.path.exists(META_MODEL_PATH):
        raise FileNotFoundError(f"Meta-modelo não encontrado em {META_MODEL_PATH}")

    # Carrega o meta-modelo e a lista de colunas esperadas
    artefatos = joblib.load(META_MODEL_PATH)
    meta_model = artefatos["model"]
    
    # Garante que a lista de colunas alvo (do modelo treinado) seja única
    loaded_feature_cols = artefatos["columns"]
    unique_feature_cols = list(dict.fromkeys(loaded_feature_cols))
    feature_cols = pd.Index(unique_feature_cols)

    print(f"Analisando o novo dataset: {caminho_csv}")
    df_completo, X_sample, y_sample = carregar_dados(caminho_csv)
    
    print("Extraindo meta-features do novo dataset...")
    meta_features = extrair_meta_features(X_sample, y_sample)

    # Prepara o DataFrame de meta-features para a previsão
    meta_df = pd.DataFrame([meta_features]).fillna(0)
    
    # Remove colunas duplicadas do DataFrame recém-criado ANTES de reindexar
    meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]

    # Reindexa para garantir que as colunas correspondam exatamente às do treinamento do meta-modelo
    meta_df = meta_df.reindex(columns=feature_cols, fill_value=0)

    # --- NOVA CORREÇÃO: Força a conversão de todas as colunas para numérico ---
    # Isso converte qualquer string que possa ter restado em NaN, que é então preenchido com 0.
    meta_df_numeric = meta_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    # --- FIM DA NOVA CORREÇÃO ---

    print("Prevendo os melhores hiperparâmetros com o meta-modelo...")
    pred = meta_model.predict(meta_df_numeric)[0]
    best_params_pred = {
        "n_estimators": int(round(pred[0])),
        "max_depth": None if int(round(pred[1])) == -1 else int(round(pred[1])),
        "min_samples_leaf": int(round(pred[2])),
    }

    print("\n===== MELHORES HIPERPARÂMETROS PREVISTOS =====")
    print(best_params_pred)

    # Treina e salva o modelo final com os parâmetros recomendados
    nome_base = os.path.splitext(os.path.basename(caminho_csv))[0]
    treinar_modelo_final(df_completo, best_params_pred, nome_base)
    
    return best_params_pred


# ====================================================
# MAIN
# ====================================================
if __name__ == "__main__":
    NOVO_DATASET = "PCA_CIC-DDoS2019.csv"  # ajuste aqui
    prever_e_treinar(NOVO_DATASET)

