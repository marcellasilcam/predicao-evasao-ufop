"""
Módulo: treinamento_modelos
Responsabilidade: Treinar modelos usando One-Hot Encoding para categorias (Cotas).

Obs:
- O RandomForest abaixo já utiliza os melhores hiperparâmetros encontrados via GridSearchCV
- A Validação Cruzada (CV) foi utilizada anteriormente para selecionar esses parâmetros
"""

import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import TransformerMixin, BaseEstimator

# --- CLASSE AUXILIAR PARA CORRIGIR O ERRO DO NAIVE BAYES ---
class DenseTransformer(TransformerMixin, BaseEstimator):
    """Converte matrizes esparsas em densas para compatibilidade com GaussianNB."""
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, "toarray") else X


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "dados_tratados.csv"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs", "modelos_salvos"))


def main():
    if not os.path.exists(PATH_DATA):
        print("Execute o preprocessamento primeiro!")
        return

    df = pd.read_csv(PATH_DATA)

    # Removemos o id_aluno e o target para as features (X)
    X = df.drop(['target_evasao', 'id_aluno'], axis=1, errors='ignore')
    y = df['target_evasao']

    # Colunas baseadas no preprocessamento
    num_cols = ['coeficiente', 'total_reprovacao_falta', 'total_reprovacao_nota', 'idade']
    cat_cols = ['sexo', 'tipo_de_cota', 'naturalidade_op']

    # --- Preprocessador ---
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # Hold-out com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ==========================================================
    # MODELOS
    # ==========================================================
    models = {
        # RandomForest com hiperparâmetros ótimos
        # Esses valores foram definidos a partir de uma etapa prévia
        # de GridSearchCV com validação cruzada estratificada (CV),
        # cujo objetivo foi maximizar o F1-score.
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        # KNN permanece como modelo comparativo
        "KNN": KNeighborsClassifier(),

        # Naive Bayes como baseline probabilístico
        "NaiveBayes": GaussianNB()
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, model in models.items():
        steps = [("preproc", preprocessor)]

        # O Naive Bayes requer dados densos
        if name == "NaiveBayes":
            steps.append(("dense", DenseTransformer()))

        steps.append(("clf", model))
        pipe = Pipeline(steps)

        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            print(f"\nModelo: {name}")
            print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred))

            joblib.dump(pipe, os.path.join(OUTPUT_DIR, f"{name.lower()}_model.joblib"))

        except Exception as e:
            print(f"\n[ERRO] Falha ao treinar o modelo {name}: {e}")


if __name__ == "__main__":
    main()
