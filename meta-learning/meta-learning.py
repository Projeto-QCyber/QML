import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pymfe.mfe import MFE
import joblib
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ====================================================
# CONFIGURAÇÕES
# ====================================================
RANDOM_STATE = 42
N_JOBS = -1
RESULTADOS_DIR = "resultados_meta"
os.makedirs(RESULTADOS_DIR, exist_ok=True)

datasets_a_processar = {
    "EdgeIIoT": "ML-EdgeIIoT-dataset-pp.csv",
    "UNSW_NB15": "UNSW_NB15_Combinado_preprocessing.csv"
}

param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [20, 40, None],
    "min_samples_leaf": [1, 5, 10],
}

# ====================================================
# FUNÇÕES AUXILIARES
# ====================================================
def carregar_dados(caminho, alvo="Attack_label"):
    df = pd.read_csv(caminho)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=[alvo])
    y = df[alvo]

    # Amostra estratificada de 20%
    X_sample, _, y_sample, _ = train_test_split(
        X, y, test_size=0.8, stratify=y, random_state=RANDOM_STATE
    )

    X_numeric = X_sample.select_dtypes(include=np.number)
    X_numeric = X_numeric.apply(pd.to_numeric, errors="coerce").fillna(0).values

    y_encoded = LabelEncoder().fit_transform(y_sample)

    return X_numeric, y_encoded


def extrair_meta_features(X, y):
    mfe = MFE(
        groups=["general", "statistical", "info-theory"],
        random_state=RANDOM_STATE,
        summary=("mean", "sd"),
    )
    mfe.fit(X, y)
    features, names = mfe.extract()
    return dict(zip(names, features))


# ====================================================
# PROCESSAMENTO DOS DATASETS
# ====================================================
experiencias = []

for nome, caminho in datasets_a_processar.items():
    print(f"\n--- Processando dataset: {nome} ---")
    start = time.time()
    try:
        X, y = carregar_dados(caminho)

        print("Extraindo meta-features...")
        meta_features = extrair_meta_features(X, y)

        print("Rodando GridSearchCV...")
        grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            param_grid_rf,
            cv=2,
            n_jobs=N_JOBS,
        )
        grid.fit(X, y)
        best_params = grid.best_params_

        experiencia = meta_features.copy()
        experiencia["target_n_estimators"] = best_params["n_estimators"]
        experiencia["target_max_depth"] = (
            -1 if best_params["max_depth"] is None else best_params["max_depth"]
        )
        experiencia["target_min_samples_leaf"] = best_params["min_samples_leaf"]
        experiencias.append(experiencia)

        # --- NOVO: SALVANDO ARTEFATOS INDIVIDUAIS PARA CADA BASE ---

        # Salvar os melhores hiperparâmetros em CSV (já existente)
        pd.DataFrame([best_params]).to_csv(
            os.path.join(RESULTADOS_DIR, f"melhores_params_{nome}.csv"), index=False
        )

        # Salvar os melhores hiperparâmetros em TXT
        txt_path = os.path.join(RESULTADOS_DIR, f"melhores_params_{nome}.txt")
        with open(txt_path, 'w') as f:
            f.write(str(best_params))
        print(f"-> Melhores parâmetros salvos em: {txt_path}")

        # Salvar o modelo final treinado para esta base
        best_model = grid.best_estimator_
        model_path = os.path.join(RESULTADOS_DIR, f"modelo_final_{nome}.joblib")
        joblib.dump(best_model, model_path)
        print(f"-> Modelo final salvo em: {model_path}")
        
        # --- FIM DA SEÇÃO NOVA ---

        print(f"Concluído em {time.time() - start:.2f}s. Best params: {best_params}")

    except Exception as e:
        print(f"[ERRO] Falha ao processar {nome}: {e}")


# ====================================================
# TREINAR META-MODELO
# ====================================================
if not experiencias:
    print("\nNenhuma experiência coletada. Encerrando.")
    exit()

meta_df = pd.DataFrame(experiencias).fillna(0)
# Remover duplicatas de colunas
meta_df = meta_df.loc[:, ~meta_df.columns.duplicated()]

target_cols = ["target_n_estimators", "target_max_depth", "target_min_samples_leaf"]
feature_cols = [c for c in meta_df.columns if c not in target_cols]

X_meta = meta_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_meta = meta_df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

print("\nTreinando Meta-Modelo...")
meta_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS)
meta_model.fit(X_meta, y_meta)

artefatos = {"model": meta_model, "columns": feature_cols}
joblib.dump(artefatos, os.path.join(RESULTADOS_DIR, "meta_modelo_pymfe.joblib"))

print("\n✅ Meta-modelo treinado e salvo em resultados_meta/meta_modelo_pymfe.joblib")