"""
Módulo: treinamento_modelos
Responsabilidade: Treinar modelos usando One-Hot Encoding para categorias (Cotas).

Obs:
- RandomForest utiliza hiperparâmetros previamente selecionados via GridSearchCV
  (código do GridSearch mantido comentado para documentação metodológica).
- XGBoost utiliza hiperparâmetros obtidos via GridSearchCV com validação cruzada,
  também mantido comentado.
- KNN é utilizado como modelo comparativo (baseline).
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# XGBoost
from xgboost import XGBClassifier  # se der erro: python -m pip install xgboost


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "dados_tratados.csv"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs", "modelos_salvos"))


def print_metrics(name: str, y_true, y_pred) -> None:
    """Imprime acurácia geral + métricas por classe."""
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    print(f"\nModelo: {name}")
    print(f"Acurácia (geral): {acc:.4f}\n")

    print("Classe 0 (não evasão)")
    print(f"  Precisão : {prec[0]:.4f}")
    print(f"  Revocação: {rec[0]:.4f}")
    print(f"  F1-Score : {f1[0]:.4f}")
    print(f"  Suporte  : {sup[0]}")

    print("\nClasse 1 (evasão)")
    print(f"  Precisão : {prec[1]:.4f}")
    print(f"  Revocação: {rec[1]:.4f}")
    print(f"  F1-Score : {f1[1]:.4f}")
    print(f"  Suporte  : {sup[1]}")


def main():
    if not os.path.exists(PATH_DATA):
        print("Execute o preprocessamento primeiro!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(PATH_DATA)

    # Variáveis preditoras e alvo
    X = df.drop(["target_evasao", "id_aluno"], axis=1, errors="ignore")
    y = df["target_evasao"]

    # Colunas definidas no preprocessamento
    num_cols = ["coeficiente", "total_reprovacao_falta", "total_reprovacao_nota", "idade"]
    cat_cols = ["sexo", "tipo_de_cota", "naturalidade_op"]

    # Preprocessador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # Divisão hold-out estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Ajuste para desbalanceamento da classe positiva
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # ==========================================================
    # 1) RANDOM FOREST (hiperparâmetros fixados)
    # ==========================================================
    rf_pipe = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=600,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="log2",
                bootstrap=True,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    print_metrics("RandomForest", y_test, y_pred_rf)
    joblib.dump(rf_pipe, os.path.join(OUTPUT_DIR, "randomforest_model.joblib"))

    # ----------------------------------------------------------
    # GridSearchCV utilizado para encontrar os hiperparâmetros
    # da Floresta Randômica (comentado para documentação)
    # ----------------------------------------------------------
    """
    param_grid_rf = {
        "clf__n_estimators": [300, 600, 900],
        "clf__max_depth": [8, 10, 12],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }

    cv_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_grid = GridSearchCV(
        estimator=rf_pipe,
        param_grid=param_grid_rf,
        scoring="f1",
        cv=cv_rf,
        n_jobs=-1,
        verbose=1
    )

    rf_grid.fit(X_train, y_train)
    print(rf_grid.best_params_)
    """

    # ==========================================================
    # 2) XGBOOST (hiperparâmetros fixados)
    # ==========================================================
    xgb_pipe = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.03,
                subsample=1.0,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    print_metrics("XGBoost", y_test, y_pred_xgb)
    joblib.dump(xgb_pipe, os.path.join(OUTPUT_DIR, "xgboost_model.joblib"))

    # ----------------------------------------------------------
    # GridSearchCV utilizado para encontrar os hiperparâmetros
    # do XGBoost (comentado para documentação)
    # ----------------------------------------------------------
    """
    param_grid_xgb = {
        "clf__n_estimators": [300, 600, 900],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.8, 0.9, 1.0],
        "clf__reg_lambda": [1.0, 2.0],
        "clf__min_child_weight": [1, 3],
        "clf__gamma": [0, 0.5],
    }

    cv_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_grid = GridSearchCV(
        estimator=xgb_pipe,
        param_grid=param_grid_xgb,
        scoring="f1",
        cv=cv_xgb,
        n_jobs=-1,
        verbose=1
    )

    xgb_grid.fit(X_train, y_train)
    print(xgb_grid.best_params_)
    """

    # ==========================================================
    # 3) KNN (baseline)
    # ==========================================================
    knn_pipe = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", KNeighborsClassifier()),
        ]
    )

    knn_pipe.fit(X_train, y_train)
    y_pred_knn = knn_pipe.predict(X_test)
    print_metrics("KNN", y_test, y_pred_knn)
    joblib.dump(knn_pipe, os.path.join(OUTPUT_DIR, "knn_model.joblib"))


if __name__ == "__main__":
    main()
