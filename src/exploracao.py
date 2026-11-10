"""Análise exploratória: taxas de falta por grupo e associações com o desfecho.

A taxa de falta de um grupo é o número de faltas dividido pelo número de
consultas daquele grupo.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def pct(valor, casas=1):
    """Formata uma proporção como porcentagem com vírgula (0.2019 -> '20,2%')."""
    return f"{100 * valor:.{casas}f}%".replace(".", ",")


def taxa_por(df, coluna):
    return df.groupby(coluna, observed=True)["NoShow"].agg(consultas="size", faltas="sum", taxa="mean")


def resumo_sms(df):
    """Números usados para discutir o SMS (hipótese H2)."""
    com_espera_minima = df[df["WaitingDays"] >= 3]
    return {
        "taxa_geral": {int(k): float(v) for k, v in df.groupby("SMS_received")["NoShow"].mean().items()},
        "menor_espera_com_sms": int(df.loc[df["SMS_received"] == 1, "WaitingDays"].min()),
        "consultas_mesmo_dia_com_sms": int(df.loc[df["WaitingDays"] == 0, "SMS_received"].sum()),
        "taxa_espera_3_dias_ou_mais": {
            int(k): float(v) for k, v in com_espera_minima.groupby("SMS_received")["NoShow"].mean().items()
        },
        "consultas_espera_3_dias_ou_mais": int(len(com_espera_minima)),
    }


def sms_por_espera(df):
    """Taxa de falta com e sem SMS dentro de cada faixa de espera.

    A comparação geral engana porque nenhuma consulta do mesmo dia recebeu SMS,
    e esse é justamente o grupo que menos falta. Separando por faixa de espera
    a comparação fica mais justa.
    """
    linhas = []
    for faixa, grupo in df.groupby("FaixaEspera", observed=True):
        sem = grupo[grupo["SMS_received"] == 0]
        com = grupo[grupo["SMS_received"] == 1]
        linhas.append({
            "FaixaEspera": faixa,
            "consultas_sem_sms": len(sem),
            "taxa_sem_sms": sem["NoShow"].mean(),
            "consultas_com_sms": len(com),
            "taxa_com_sms": com["NoShow"].mean() if len(com) else np.nan,
        })
    return pd.DataFrame(linhas).set_index("FaixaEspera")


ATRIBUTOS_BINARIOS = [
    ("Gender_M", "Sexo masculino"),
    ("Scholarship", "Bolsa Família"),
    ("Hipertension", "Hipertensão"),
    ("Diabetes", "Diabetes"),
    ("Alcoholism", "Alcoolismo"),
    ("Handcap", "Deficiência (Handcap > 0)"),
    ("SMS_received", "SMS recebido"),
]


def associacoes(df):
    """Teste qui-quadrado e V de Cramér de cada atributo binário com a falta (Tabela 5).

    Como as tabelas são 2x2, o V de Cramér é a raiz de qui-quadrado / n.
    Os testes são só exploratórios, porque o mesmo paciente aparece em várias
    consultas e as observações não são totalmente independentes.
    """
    linhas = []
    for coluna, nome in ATRIBUTOS_BINARIOS:
        tem = (df[coluna] > 0).astype(int)
        tabela = pd.crosstab(tem, df["NoShow"])
        qui2, p_valor, _, _ = chi2_contingency(tabela)
        linhas.append({
            "Atributo": nome,
            "Consultas com o atributo": int(tem.sum()),
            "Taxa com": df.loc[tem == 1, "NoShow"].mean(),
            "Taxa sem": df.loc[tem == 0, "NoShow"].mean(),
            "Qui-quadrado": qui2,
            "p-valor": p_valor,
            "V de Cramér": np.sqrt(qui2 / tabela.values.sum()),
        })
    return pd.DataFrame(linhas)


def hipertensao_por_idade(df):
    """Taxa de falta com e sem hipertensão dentro de cada faixa etária.

    A hipertensão é mais comum entre pessoas mais velhas, que também faltam
    menos. Comparar dentro da mesma faixa mostra quanto da diferença vem da idade.
    """
    tabela = df.pivot_table(index="FaixaIdade", columns="Hipertension", values="NoShow",
                            aggfunc="mean", observed=True)
    tabela.columns = ["Taxa sem hipertensão", "Taxa com hipertensão"]
    tabela["Prevalência de hipertensão"] = df.groupby("FaixaIdade", observed=True)["Hipertension"].mean()
    return tabela
