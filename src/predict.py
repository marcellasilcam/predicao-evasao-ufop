# predict.py
"""
Módulo: predict
Responsabilidade: Carregar o pipeline final (RandomForest) e, dado um aluno de entrada,
retornar a probabilidade (%) de evasão e uma classificação por limiar.

Compatível com o pipeline salvo em:
../outputs/modelos_salvos/randomforest_model.joblib

Entrada esperada (features):
- sexo: "M" ou "F" (string)
- idade: int
- tipo_de_cota: ex "AC" (string)
- coeficiente: float
- naturalidade_op: 0 ou 1 (int)  -> 1 se nasceu em Ouro Preto, 0 caso contrário
- total_reprovacao_nota: int
- total_reprovacao_falta: int
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import joblib
import pandas as pd


def _default_model_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(base_dir, "..", "outputs", "modelos_salvos", "randomforest_model.joblib")
    )


REQUIRED_FIELDS = [
    "sexo",
    "idade",
    "tipo_de_cota",
    "coeficiente",
    "naturalidade_op",
    "total_reprovacao_nota",
    "total_reprovacao_falta",
]


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em: {model_path}\n"
            f"Dica: verifique se você já rodou treinamento_modelos.py e se o arquivo foi salvo."
        )
    return joblib.load(model_path)


def validate_and_normalize_input(aluno: Dict[str, Any]) -> Dict[str, Any]:
    # Checar campos obrigatórios
    missing = [f for f in REQUIRED_FIELDS if f not in aluno]
    if missing:
        raise ValueError(f"Campos faltando no aluno: {missing}")

    # Normalizações leves (mantendo compatibilidade com preprocessamento)
    aluno_norm = dict(aluno)

    # sexo
    aluno_norm["sexo"] = str(aluno_norm["sexo"]).strip().upper()
    if aluno_norm["sexo"] not in ("M", "F"):
        # não impede, mas alerta: OneHotEncoder handle_unknown='ignore' aguenta valores novos
        pass

    # tipo_de_cota
    aluno_norm["tipo_de_cota"] = str(aluno_norm["tipo_de_cota"]).strip().upper()

    # naturalidade_op: garantir 0/1 int
    try:
        aluno_norm["naturalidade_op"] = int(aluno_norm["naturalidade_op"])
    except Exception:
        raise ValueError("naturalidade_op deve ser 0 ou 1 (inteiro).")

    if aluno_norm["naturalidade_op"] not in (0, 1):
        raise ValueError("naturalidade_op deve ser 0 ou 1.")

    # casts numéricos
    try:
        aluno_norm["idade"] = int(aluno_norm["idade"])
        aluno_norm["total_reprovacao_nota"] = int(aluno_norm["total_reprovacao_nota"])
        aluno_norm["total_reprovacao_falta"] = int(aluno_norm["total_reprovacao_falta"])
        aluno_norm["coeficiente"] = float(aluno_norm["coeficiente"])
    except Exception as e:
        raise ValueError(f"Erro ao converter tipos numéricos: {e}")

    return aluno_norm


def predict_risk(model, aluno: Dict[str, Any]) -> float:
    """
    Retorna probabilidade de evasão (classe 1) como float entre 0 e 1.
    """
    aluno_norm = validate_and_normalize_input(aluno)
    df = pd.DataFrame([aluno_norm], columns=REQUIRED_FIELDS)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("O modelo carregado não suporta predict_proba().")

    proba = model.predict_proba(df)[0][1]  # classe 1 = evasão
    return float(proba)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Predição de risco de evasão (RandomForest).")

    parser.add_argument("--model", type=str, default=_default_model_path(),
                        help="Caminho do modelo .joblib (padrão: randomforest_model.joblib).")

    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Limiar para classificar como 'alto risco' (padrão: 0.70).")

    # Entrada via JSON
    parser.add_argument("--json", type=str, default=None,
                        help="Caminho para um arquivo JSON com os dados do aluno.")

    # Entrada via CLI (se não usar JSON)
    parser.add_argument("--sexo", type=str, default=None, help="M ou F")
    parser.add_argument("--idade", type=int, default=None, help="Idade em anos")
    parser.add_argument("--tipo_de_cota", type=str, default=None, help="Ex: AC")
    parser.add_argument("--coeficiente", type=float, default=None, help="CR/coeficiente")
    parser.add_argument("--naturalidade_op", type=int, default=None, help="1 se Ouro Preto, 0 caso contrário")
    parser.add_argument("--total_reprovacao_nota", type=int, default=None, help="Total reprovação por nota")
    parser.add_argument("--total_reprovacao_falta", type=int, default=None, help="Total reprovação por falta")

    return parser.parse_args()


def main():
    args = parse_args()

    # Monta aluno
    if args.json:
        aluno = read_json(args.json)
    else:
        aluno = {
            "sexo": args.sexo,
            "idade": args.idade,
            "tipo_de_cota": args.tipo_de_cota,
            "coeficiente": args.coeficiente,
            "naturalidade_op": args.naturalidade_op,
            "total_reprovacao_nota": args.total_reprovacao_nota,
            "total_reprovacao_falta": args.total_reprovacao_falta,
        }

    # Valida se veio tudo quando é CLI (sem JSON)
    if not args.json and any(v is None for v in aluno.values()):
        print("Erro: faltam argumentos. Use --json aluno.json OU passe todos os campos via CLI.\n")
        print("Exemplo CLI:")
        print("  python predict.py --sexo F --idade 21 --tipo_de_cota AC --coeficiente 7.3 "
              "--naturalidade_op 1 --total_reprovacao_nota 2 --total_reprovacao_falta 1")
        sys.exit(1)

    model = load_model(args.model)
    proba = predict_risk(model, aluno)

    risk_pct = proba * 100.0
    label = "ALTO RISCO" if proba >= args.threshold else "BAIXO/MODERADO"

    print(f"Probabilidade de evasão: {risk_pct:.2f}%")
    print(f"Classificação (limiar={args.threshold:.2f}): {label}")


if __name__ == "__main__":
    main()
