"""Configurações usadas em todo o projeto: caminhos, semente e listas de atributos."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ARQUIVO_CSV = RAIZ / "data" / "KaggleV2-May-2016.csv"
PASTA_FIGURAS = RAIZ / "figures"
PASTA_RESULTADOS = RAIZ / "results"

# Semente fixa para que a divisão dos dados e os modelos deem sempre o mesmo resultado.
SEED = 42

# SHA-256 do arquivo baixado do Kaggle e usado no artigo. Serve para conferir
# se a base usada numa nova execução é exatamente a mesma.
SHA256_ESPERADO = "9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d"

# Faixas usadas apenas nas tabelas e gráficos descritivos.
# Os modelos recebem a idade e o tempo de espera como números contínuos.
FAIXAS_IDADE = [-1, 12, 18, 30, 45, 60, 75, 200]
ROTULOS_IDADE = ["0–12", "13–18", "19–30", "31–45", "46–60", "61–75", "76+"]

FAIXAS_ESPERA = [-1, 0, 3, 7, 14, 30, 60, 200]
ROTULOS_ESPERA = ["0", "1–3", "4–7", "8–14", "15–30", "31–60", "61+"]

# Experimento A (principal): só o que já se sabe no momento do agendamento.
ATRIBUTOS_A = [
    "Gender_M",
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "WaitingDays",
]

# Experimento B (comparação): inclui o SMS, que só existe depois do envio.
ATRIBUTOS_B = ATRIBUTOS_A + ["SMS_received"]

# Atributos que passam por padronização na Regressão Logística e no MLP.
NUMERICOS = ["Age", "WaitingDays", "Handcap"]
