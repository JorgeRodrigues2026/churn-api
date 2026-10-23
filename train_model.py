"""
# Fase 1 — Preparação e Serialização do Modelo (Pipeline Offline)

Problema de negócio
-------------------
Prever a probabilidade de um cliente de telecom cancelar o contrato (churn),
para que o time de retenção aja ANTES do cancelamento.

Base de dados
-------------
IBM Telco Customer Churn (7.043 clientes, 21 colunas) — base pública, também
disponível no Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
O CSV fica em data/telco_churn.csv (caminho relativo, nunca absoluto).

Como executar (a partir da raiz do projeto):
    python training/train_model.py

Saídas:
    models/churn_pipeline.joblib  -> pipeline completo (pré-processamento + modelo)
    models/metrics.json           -> relatório de métricas (validação e teste)
"""

# ## 1. Imports e caminhos relativos
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Caminhos SEMPRE relativos à raiz do projeto (o avaliador roda em outra máquina)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42  # reprodutibilidade
MODEL_VERSION = "1.0.0"

# ## 2. Definição das features
# Decisão técnica: customerID é só um identificador (não generaliza) e é removido.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


# ## 3. Carga e limpeza
def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # TotalCharges vem como texto e tem 11 valores em branco (clientes com tenure=0).
    # Convertemos para número; os vazios viram NaN e serão IMPUTADOS dentro do pipeline
    # (imputar aqui, antes do split, vazaria informação do teste para o treino).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen é 0/1, mas semanticamente é categórica -> padronizamos como "Yes"/"No"
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


# ## 4. Pré-processamento (imputação, codificação categórica, escalonamento)
def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": categoria nova em produção não derruba a API
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


# ## 5. Candidatos
# Decisão técnica: comparamos um modelo linear interpretável (Regressão Logística,
# já vista no curso) com um ensemble não linear (Random Forest).
# class_weight="balanced" compensa o desbalanceamento (~26,5% de churn).
def build_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=8,
                                             min_samples_leaf=5, class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1_score": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main() -> None:
    # ## 6. Divisão treino / validação / teste (60 / 20 / 20, estratificada)
    X, y = load_data()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)
    print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    print(f"Taxa de churn no treino: {y_train.mean():.2%}\n")

    # ## 7. Seleção do modelo pela ROC-AUC na VALIDAÇÃO
    # Decisão técnica: ROC-AUC mede a capacidade de ordenar clientes por risco,
    # independente do limiar — é o que o time de retenção usa para priorizar.
    val_results = {}
    fitted = {}
    for name, pipe in build_candidates().items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        val_results[name] = evaluate(pipe, X_val, y_val)
        print(f"[validação] {name}: {val_results[name]}")

    best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])
    print(f"\nModelo escolhido: {best_name}")

    # ## 8. Re-treino em treino+validação e avaliação final no TESTE (nunca visto)
    best = build_candidates()[best_name]
    best.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_metrics = evaluate(best, X_test, y_test)
    test_pred = best.predict(X_test)
    print(f"[teste] {best_name}: {test_metrics}\n")
    print(classification_report(y_test, test_pred, target_names=["Fica", "Churn"]))

    # ## 9. Serialização do artefato + relatório de métricas
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "pipeline": best,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_PATH)

    report = {
        "dataset": "IBM Telco Customer Churn",
        "n_samples": int(len(X)),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "selected_model": best_name,
        "model_version": MODEL_VERSION,
        "validation_metrics": val_results,
        "test_metrics": test_metrics,
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo salvo em: {MODEL_PATH.relative_to(ROOT_DIR)}")
    print(f"Métricas salvas em: {METRICS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
