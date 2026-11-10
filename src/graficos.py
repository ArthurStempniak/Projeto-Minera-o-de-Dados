"""Gráficos do trabalho.

As funções grafico_desfecho, grafico_taxa, grafico_metricas e grafico_roc
geram as Figuras 1 a 5 do artigo. As demais são gráficos de apoio que ajudam
a entender os resultados, mas não entraram no texto.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from .config import PASTA_FIGURAS
from .exploracao import pct
from .modelagem import MODELOS

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE = "#1baf7a"
ROXO = "#6250d6"
CINZA = "#52514e"
TEXTO = "#0b0b0b"

CORES_MODELOS = {"Regressão Logística": AZUL, "Árvore de Decisão": LARANJA, "MLP": VERDE}
CORES_DESFECHO = {"Compareceu": AZUL, "Faltou": LARANJA}


def configurar_estilo():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "grid.color": "#e4e3df",
        "axes.edgecolor": "#c3c2b7",
        "axes.titleweight": "bold",
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO,
        "xtick.color": CINZA,
        "ytick.color": CINZA,
    })


def salvar(fig, nome, pasta=PASTA_FIGURAS):
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome
    fig.savefig(caminho)
    return caminho


def _virgula(casas):
    """Formatador de eixo com vírgula decimal."""
    return mtick.FuncFormatter(lambda x, _: f"{x:.{casas}f}".replace(".", ","))


def _milhar(valor):
    return f"{int(valor):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Figuras do artigo
# ---------------------------------------------------------------------------

def grafico_desfecho(df):
    """Figura 1 - Distribuição do desfecho das consultas."""
    contagem = df["NoShow"].map({0: "Compareceu", 1: "Faltou"}).value_counts().reindex(["Compareceu", "Faltou"])
    dados = contagem.rename_axis("Desfecho").reset_index(name="Consultas")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(data=dados, x="Desfecho", y="Consultas", hue="Desfecho", palette=CORES_DESFECHO,
                legend=False, width=0.6, ax=ax)
    for barra, valor in zip(ax.patches, dados["Consultas"]):
        ax.annotate(f"{_milhar(valor)} ({pct(valor / dados['Consultas'].sum())})",
                    (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_title("Distribuição do desfecho das consultas")
    ax.set_xlabel("")
    ax.set_ylabel("Número de consultas")
    ax.set_ylim(0, contagem.max() * 1.12)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: _milhar(x)))
    fig.tight_layout()
    return fig


def grafico_taxa(tabela, taxa_geral, titulo, rotulo_x, cor=AZUL, posicao_legenda="upper left"):
    """Figuras 2 e 3 - Taxa de falta por faixa (espera ou idade)."""
    dados = tabela.reset_index()
    coluna = dados.columns[0]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(data=dados, x=coluna, y="taxa", color=cor, width=0.7, ax=ax)
    for barra, valor in zip(ax.patches, dados["taxa"]):
        ax.annotate(pct(valor), (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.axhline(taxa_geral, color=LARANJA, linestyle="--", linewidth=1.2,
               label=f"Taxa geral: {pct(taxa_geral)}")
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Taxa de falta")
    ax.set_ylim(0, dados["taxa"].max() * 1.18)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(loc=posicao_legenda, frameon=False)
    fig.tight_layout()
    return fig


def grafico_metricas(metricas, titulo="Desempenho dos modelos no teste - experimento A"):
    """Figura 4 - Precisão, recall, F1 e ROC-AUC de cada modelo."""
    nomes = ["Precisão", "Recall", "F1-score", "ROC-AUC"]
    dados = (metricas.loc[MODELOS, nomes].rename_axis("Modelo").reset_index()
             .melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.barplot(data=dados, x="Modelo", y="Valor", hue="Métrica",
                palette=[AZUL, LARANJA, VERDE, ROXO], width=0.8, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{b.get_height():.2f}".replace(".", ",") for b in container],
                     padding=2, fontsize=7, color=CINZA)
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(_virgula(1))
    ax.legend(ncol=2, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def grafico_roc(modelos, teste, atributos, titulo="Curvas ROC - experimento A"):
    """Figura 5 - Curvas ROC no conjunto de teste."""
    y = teste["NoShow"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome in MODELOS:
        prob = modelos[nome].predict_proba(teste[atributos])[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, linewidth=2, color=CORES_MODELOS[nome],
                label=f"{nome} (AUC = {auc:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#8c8b86", label="Aleatório (AUC = 0,500)")
    ax.set_title(titulo)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.xaxis.set_major_formatter(_virgula(1))
    ax.yaxis.set_major_formatter(_virgula(1))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gráficos de apoio
# ---------------------------------------------------------------------------

def grafico_sms_por_espera(tabela_sms):
    """Taxa de falta com e sem SMS dentro de cada faixa de espera."""
    faixas = tabela_sms.index.astype(str)
    dados = pd.concat([
        pd.DataFrame({"Espera": faixas, "Grupo": "Sem SMS", "Taxa": tabela_sms["taxa_sem_sms"].values}),
        pd.DataFrame({"Espera": faixas, "Grupo": "Com SMS", "Taxa": tabela_sms["taxa_com_sms"].values}),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=dados, x="Espera", y="Taxa", hue="Grupo",
                palette={"Sem SMS": AZUL, "Com SMS": LARANJA}, width=0.75, ax=ax)
    for i, valor in enumerate(tabela_sms["taxa_com_sms"]):
        if np.isnan(valor):
            ax.text(i + 0.2, 0.01, "sem envio", rotation=90, fontsize=7.5, color=CINZA, ha="center", va="bottom")
    ax.set_title("Taxa de falta com e sem SMS, por tempo de espera")
    ax.set_xlabel("Dias entre agendamento e consulta")
    ax.set_ylabel("Taxa de falta")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(title="", loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def grafico_matrizes_confusao(metricas, titulo="Matrizes de confusão no teste - experimento A"):
    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, nome in zip(eixos, MODELOS):
        linha = metricas.loc[nome]
        matriz = np.array([[linha["VN"], linha["FP"]], [linha["FN"], linha["VP"]]])
        anotacoes = [[f"{_milhar(matriz[i, j])}\n({pct(matriz[i, j] / matriz[i].sum())})" for j in range(2)]
                     for i in range(2)]
        sns.heatmap(matriz, annot=anotacoes, fmt="", cmap="Blues", cbar=False, linewidths=2, linecolor="white",
                    xticklabels=["Compareceu", "Faltou"], yticklabels=["Compareceu", "Faltou"],
                    annot_kws={"fontsize": 8.5}, ax=ax)
        ax.set_title(nome, fontsize=10)
        ax.set_xlabel("Classe prevista")
    eixos[0].set_ylabel("Classe real")
    fig.suptitle(titulo, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def grafico_histograma(df, coluna, titulo, rotulo_x, limite, escala_log=False):
    dados = df.assign(Desfecho=df["NoShow"].map({0: "Compareceu", 1: "Faltou"}))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.histplot(data=dados, x=coluna, hue="Desfecho", bins=np.arange(0, limite + 1, 5), stat="density",
                 common_norm=False, element="step", fill=False, linewidth=2, palette=CORES_DESFECHO, ax=ax)
    if escala_log:
        ax.set_yscale("log")
    ax.set_title(titulo)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("Densidade" + (" (escala log)" if escala_log else ""))
    sns.move_legend(ax, "upper right", title="", frameon=False)
    fig.tight_layout()
    return fig
