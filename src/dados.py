"""Carga da base, verificação de qualidade e preparação dos dados."""

import hashlib

import numpy as np
import pandas as pd

from .config import (ARQUIVO_CSV, FAIXAS_ESPERA, FAIXAS_IDADE, ROTULOS_ESPERA,
                     ROTULOS_IDADE, SHA256_ESPERADO)


def calcular_sha256(caminho=ARQUIVO_CSV):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def carregar_base(caminho=ARQUIVO_CSV):
    """Lê o CSV original e avisa se o arquivo não for o mesmo usado no artigo."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Baixe a base no Kaggle (ver data/README.md) e coloque o CSV na pasta data/."
        )
    sha = calcular_sha256(caminho)
    if sha != SHA256_ESPERADO:
        print("Atenção: o CSV é diferente do usado no artigo. Os números podem não bater.")
    return pd.read_csv(caminho)


def dias_de_espera(bruto):
    """Dias entre o agendamento e a consulta (0 = consulta no mesmo dia).

    AppointmentDay sempre vem com hora 00:00, por isso as duas datas são
    comparadas só pelo dia, sem considerar o horário do agendamento.
    """
    agendamento = pd.to_datetime(bruto["ScheduledDay"]).dt.normalize()
    consulta = pd.to_datetime(bruto["AppointmentDay"]).dt.normalize()
    return (consulta - agendamento).dt.days


def resumo_base(bruto):
    agendamento = pd.to_datetime(bruto["ScheduledDay"])
    consulta = pd.to_datetime(bruto["AppointmentDay"])
    return {
        "linhas": int(bruto.shape[0]),
        "colunas": int(bruto.shape[1]),
        "pacientes": int(bruto["PatientId"].nunique()),
        "bairros": int(bruto["Neighbourhood"].nunique()),
        "agendamento_min": str(agendamento.min().date()),
        "agendamento_max": str(agendamento.max().date()),
        "consulta_min": str(consulta.min().date()),
        "consulta_max": str(consulta.max().date()),
        "consulta_tem_horario": bool((consulta.dt.hour + consulta.dt.minute).any()),
        "valores_handcap": {int(k): int(v) for k, v in bruto["Handcap"].value_counts().sort_index().items()},
    }


def verificar_qualidade(bruto):
    """Monta a tabela de verificações de qualidade (Tabela 1 do artigo)."""
    espera = dias_de_espera(bruto)
    consultas_por_paciente = bruto["PatientId"].value_counts()
    repetidos = consultas_por_paciente[consultas_por_paciente > 1]

    verificacoes = [
        ("Valores ausentes", bruto.isna().sum().sum(), "Nenhuma imputação necessária"),
        ("Linhas duplicadas completas", bruto.duplicated().sum(), "Nenhuma remoção"),
        ("AppointmentID repetido", bruto["AppointmentID"].duplicated().sum(), "Nenhuma remoção"),
        ("Duplicadas sem AppointmentID", bruto.drop(columns="AppointmentID").duplicated().sum(),
         "Mantidas como consultas distintas"),
        ("Pacientes com mais de uma consulta", len(repetidos), "Mantidos; agrupados na validação"),
        ("Consultas desses pacientes", repetidos.sum(), "Mantidas"),
        ("Idade negativa", (bruto["Age"] < 0).sum(), "Removida"),
        ("Idade igual a zero", (bruto["Age"] == 0).sum(), "Mantida"),
        ("Idade acima de 100 anos", (bruto["Age"] > 100).sum(), "Mantida e registrada como possível anomalia"),
        ("Consultas anteriores ao agendamento", (espera < 0).sum(), "Removidas"),
        ("Handcap com códigos 2, 3 ou 4", (bruto["Handcap"] > 1).sum(), "Mantidos; significado não documentado"),
    ]
    tabela = pd.DataFrame(verificacoes, columns=["Verificação", "Quantidade", "Tratamento"])
    tabela["Quantidade"] = tabela["Quantidade"].astype(int)
    return tabela


def registros_invalidos(bruto):
    """Lista os registros impossíveis: idade negativa e consulta antes do agendamento."""
    espera = dias_de_espera(bruto)
    invalidos = bruto[(bruto["Age"] < 0) | (espera < 0)].copy()
    invalidos["WaitingDays"] = espera[invalidos.index]
    invalidos["Motivo"] = np.where(invalidos["Age"] < 0, "Idade negativa (-1)",
                                   "Consulta anterior ao agendamento")
    colunas = ["AppointmentID", "ScheduledDay", "AppointmentDay", "Age", "WaitingDays", "No-show", "Motivo"]
    return invalidos[colunas]


def preparar_dados(bruto):
    """Remove os registros inválidos e cria as variáveis usadas no estudo.

    NoShow      1 = faltou, 0 = compareceu (variável-alvo)
    Gender_M    1 = masculino, 0 = feminino
    WaitingDays dias entre agendamento e consulta
    FaixaIdade e FaixaEspera servem só para a análise descritiva.
    """
    espera = dias_de_espera(bruto)
    df = bruto[(bruto["Age"] >= 0) & (espera >= 0)].copy()
    df["WaitingDays"] = espera[df.index].astype(int)
    df["NoShow"] = (df["No-show"] == "Yes").astype(int)
    df["Gender_M"] = (df["Gender"] == "M").astype(int)
    df["FaixaIdade"] = pd.cut(df["Age"], FAIXAS_IDADE, labels=ROTULOS_IDADE)
    df["FaixaEspera"] = pd.cut(df["WaitingDays"], FAIXAS_ESPERA, labels=ROTULOS_ESPERA)
    return df
