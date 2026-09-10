"""
Módulo: preprocessamento
Responsabilidade: Carregar os dados brutos da pasta data/raw,
aplicar limpeza, engenharia de features e salvar o dataset 
final pronto para os modelos em data/processed.
"""

import pandas as pd
import os
import time
import unicodedata
from datetime import datetime

# --- FUNÇÃO AUXILIAR PARA REMOVER ACENTOS ---
def remover_acentos(texto):
    """Remove acentos e caracteres especiais de uma string."""
    if not isinstance(texto, str):
        return texto
    return "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).replace('ç', 'c').replace('Ç', 'C').upper()

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume que a estrutura é TCC/src/preprocessamento.py
PATH_EXCEL = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw", "dados_ufop.xlsx"))
PATH_OUT_PROCESSED = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "dados_tratados.csv"))

def contar_reprovacoes(disciplinas_str: str) -> int:
    """Converte string de disciplinas '[COD1, COD2]' em contagem inteira."""
    if not isinstance(disciplinas_str, str) or not disciplinas_str or disciplinas_str == "[]" or disciplinas_str == "":
        return 0
    reprovacoes = disciplinas_str.strip('[]').split(', ')
    return len([r for r in reprovacoes if r.strip()])

def processar_e_salvar():
    """Executa todo o pipeline de limpeza e preparação de dados."""
    print(f"Lendo dados de: {os.path.abspath(PATH_EXCEL)}")
    
    if not os.path.exists(PATH_EXCEL):
        print(f"[ERRO] Arquivo não encontrado: {PATH_EXCEL}")
        return

    try:
        # 1. Carregamento
        df = pd.read_excel(PATH_EXCEL)
        
        # 2. Padronização de nomes de colunas (Maiúsculas, sem acentos, sem espaços)
        df.columns = [remover_acentos(c).replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]

        print("  -> Iniciando engenharia de features...")

        # --- Identificação do ID do Aluno ---
        col_id = None
        for c in ['ID_ALUNO', 'MATRICULA', 'ID']:
            if c in df.columns:
                col_id = c
                break
        
        if col_id:
            df['id_aluno'] = df[col_id]
        else:
            df['id_aluno'] = range(1, len(df) + 1)

        # --- Lógica para Tipo de Cota ---
        col_cota = None
        for c in ['TIPO_DE_COTA', 'MODALIDADE_CONCORRENCIA', 'COTA']:
            if c in df.columns:
                col_cota = c
                break
        
        if col_cota:
            df['tipo_de_cota'] = df[col_cota].fillna('AC').apply(remover_acentos)
        else:
            df['tipo_de_cota'] = 'AC'

        # --- Tratamento de Outras Features ---
        
        # Sexo
        col_sexo = 'SEXO' if 'SEXO' in df.columns else 'SEXO'
        df['sexo'] = df[col_sexo].fillna('M').str.upper()
        
        # Idade
        if 'DATA_NASCIMENTO' in df.columns:
            df['DATA_NASCIMENTO'] = pd.to_datetime(df['DATA_NASCIMENTO'], errors='coerce')
            ano_atual = datetime.now().year
            df['idade'] = df['DATA_NASCIMENTO'].apply(lambda x: ano_atual - x.year if pd.notnull(x) else 20)
        elif 'IDADE' in df.columns:
            df['idade'] = pd.to_numeric(df['IDADE'], errors='coerce').fillna(20)
        
        # NATURALIDADE OP (Nova Lógica Solicitada)
        # Procura por variações de "CIDADE NASCIMENTO" ou "NATURALIDADE"
        col_cidade_nascto = None
        for c in ['CIDADE_NASCIMENTO', 'NATURALIDADE', 'NATURALIDADE_CIDADE']:
            if c in df.columns:
                col_cidade_nascto = c
                break
        
        if col_cidade_nascto:
            df['naturalidade_op'] = df[col_cidade_nascto].apply(
                lambda x: 1 if remover_acentos(str(x)).strip() == "OURO PRETO" else 0
            )
        else:
            df['naturalidade_op'] = 0

        # Desempenho Acadêmico
        col_cr = 'CR' if 'CR' in df.columns else 'COEFICIENTE'
        df['coeficiente'] = pd.to_numeric(df[col_cr], errors='coerce').fillna(0)

        # Reprovações
        if 'REPROVACAO_POR_FALTA' in df.columns:
            df['total_reprovacao_falta'] = df['REPROVACAO_POR_FALTA'].apply(contar_reprovacoes)
        else:
            df['total_reprovacao_falta'] = 0

        if 'REPROVACAO_POR_NOTA' in df.columns:
            df['total_reprovacao_nota'] = df['REPROVACAO_POR_NOTA'].apply(contar_reprovacoes)
        else:
            df['total_reprovacao_nota'] = 0

        # 4. Target (Evasão) baseado no ANO EVASAO
        col_ano_evasao = None
        for c in df.columns:
            if 'ANO' in c and 'EVASAO' in c:
                col_ano_evasao = c
                break

        if not col_ano_evasao:
            raise ValueError("Coluna de ANO EVASAO não encontrada no dataset.")

        df['target_evasao'] = df[col_ano_evasao].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip().lower() == 'null' else 1
        )

        # 5. Seleção Final de Colunas para o Treino
        cols_finais = [
            'id_aluno', 'sexo', 'idade', 'tipo_de_cota', 'coeficiente', 
            'naturalidade_op', 'total_reprovacao_nota', 
            'total_reprovacao_falta', 'target_evasao'
        ]
        
        df_model = df[[c for c in cols_finais if c in df.columns]].copy()

        # 6. Salvar CSV Processado
        os.makedirs(os.path.dirname(PATH_OUT_PROCESSED), exist_ok=True)
        df_model.to_csv(PATH_OUT_PROCESSED, index=False)
        
        print(f"Sucesso! Dataset processado com {df_model.shape[0]} registros.")
        print(f"Arquivo salvo em: {PATH_OUT_PROCESSED}")

    except Exception as e:
        print(f"[ERRO CRÍTICO NO PRÉ-PROCESSAMENTO]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    processar_e_salvar()