"""Executa o trabalho completo, na ordem das etapas do CRISP-DM.

    python executar_analise.py

As figuras são gravadas em figures/ e as tabelas e métricas em results/.
A execução completa leva entre 10 e 15 minutos, quase todo esse tempo na
busca de hiperparâmetros do MLP.
"""

import json
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn

from src import avaliacao, dados, exploracao, graficos, modelagem
from src.config import ARQUIVO_CSV, ATRIBUTOS_A, ATRIBUTOS_B, PASTA_FIGURAS, PASTA_RESULTADOS
from src.exploracao import pct


def gravar_csv(tabela, nome, **kwargs):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela.round(4).to_csv(PASTA_RESULTADOS / nome, encoding="utf-8-sig", **kwargs)


def gravar_figura(fig, nome):
    graficos.salvar(fig, nome)
    plt.close(fig)


def para_json(valor):
    """Converte tipos do NumPy para tipos que o json consegue gravar."""
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    raise TypeError(f"Tipo não suportado: {type(valor)}")


def metricas_para_dict(metricas):
    return {
        modelo: {k: (float(v) if k in modelagem.COLUNAS_METRICAS else int(v)) for k, v in linha.items()}
        for modelo, linha in metricas.iterrows()
    }


def main():
    graficos.configurar_estilo()
    print(f"Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} "
          f"| matplotlib {matplotlib.__version__} | seaborn {seaborn.__version__}\n")

    # 1. Entendimento dos dados -------------------------------------------------
    print("1. Entendimento dos dados")
    bruto = dados.carregar_base()
    resumo = dados.resumo_base(bruto)
    print(f"  {resumo['linhas']} consultas, {resumo['colunas']} colunas, {resumo['pacientes']} pacientes")
    print(f"  Consultas de {resumo['consulta_min']} a {resumo['consulta_max']}; "
          f"agendamentos de {resumo['agendamento_min']} a {resumo['agendamento_max']}")

    qualidade = dados.verificar_qualidade(bruto)
    invalidos = dados.registros_invalidos(bruto)
    gravar_csv(qualidade, "tabela1_qualidade.csv", index=False)
    gravar_csv(invalidos, "registros_removidos.csv", index=False)
    print(f"  Registros inválidos encontrados: {len(invalidos)}")

    # 2. Preparação dos dados ---------------------------------------------------
    print("\n2. Preparação dos dados")
    df = dados.preparar_dados(bruto)
    taxa_geral = df["NoShow"].mean()
    print(f"  Após a limpeza: {len(df)} consultas e {df['PatientId'].nunique()} pacientes")
    print(f"  Faltas: {int(df['NoShow'].sum())} ({pct(taxa_geral)})")

    # 3. Análise exploratória ---------------------------------------------------
    print("\n3. Análise exploratória")
    por_espera = exploracao.taxa_por(df, "FaixaEspera")
    por_idade = exploracao.taxa_por(df, "FaixaIdade")
    sms = exploracao.resumo_sms(df)
    sms_espera = exploracao.sms_por_espera(df)
    assoc = exploracao.associacoes(df)
    hipertensao = exploracao.hipertensao_por_idade(df)

    gravar_csv(por_espera, "tabela4_falta_por_espera.csv")
    gravar_csv(por_idade, "falta_por_idade.csv")
    gravar_csv(sms_espera, "sms_por_espera.csv")
    gravar_csv(assoc, "tabela5_associacoes.csv", index=False)
    gravar_csv(hipertensao, "hipertensao_por_idade.csv")

    print(f"  Falta no mesmo dia: {pct(por_espera.loc['0', 'taxa'])} | "
          f"maior taxa por espera: {pct(por_espera['taxa'].max())} ({por_espera['taxa'].idxmax()} dias)")
    print(f"  SMS: {pct(sms['taxa_geral'][1])} com e {pct(sms['taxa_geral'][0])} sem "
          f"(menor espera com SMS: {sms['menor_espera_com_sms']} dias)")

    gravar_figura(graficos.grafico_desfecho(df), "figura1_distribuicao_desfecho.png")
    gravar_figura(graficos.grafico_taxa(por_espera, taxa_geral, "Taxa de não comparecimento por tempo de espera",
                                        "Dias entre agendamento e consulta"), "figura2_falta_por_espera.png")
    gravar_figura(graficos.grafico_taxa(por_idade, taxa_geral, "Taxa de não comparecimento por faixa etária",
                                        "Faixa etária (anos)", cor=graficos.VERDE,
                                        posicao_legenda="upper right"), "figura3_falta_por_idade.png")
    gravar_figura(graficos.grafico_sms_por_espera(sms_espera), "apoio_sms_por_espera.png")
    gravar_figura(graficos.grafico_histograma(df, "WaitingDays", "Distribuição do tempo de espera por desfecho",
                                              "Dias entre agendamento e consulta", 180, escala_log=True),
                  "apoio_histograma_espera.png")
    gravar_figura(graficos.grafico_histograma(df, "Age", "Distribuição da idade por desfecho",
                                              "Idade (anos)", 120), "apoio_histograma_idade.png")

    # 4. Desenho experimental ---------------------------------------------------
    print("\n4. Desenho experimental")
    idx_treino_linhas, idx_teste_linhas = modelagem.dividir_por_linhas(df)
    vazamento = modelagem.medir_vazamento(df, idx_treino_linhas, idx_teste_linhas)
    print(f"  Divisão por linhas: {vazamento['consultas_teste_com_paciente_no_treino']} de "
          f"{vazamento['consultas_teste']} consultas do teste ({pct(vazamento['proporcao'])}) "
          "são de pacientes que também estão no treino")

    treino, teste = modelagem.dividir_por_paciente(df)
    divisao = modelagem.resumo_divisao(treino, teste)
    gravar_csv(pd.DataFrame({
        "Consultas": [divisao["treino_consultas"], divisao["teste_consultas"]],
        "Pacientes": [divisao["treino_pacientes"], divisao["teste_pacientes"]],
        "Taxa de falta": [divisao["taxa_falta_treino"], divisao["taxa_falta_teste"]],
    }, index=["Treino", "Teste"]), "tabela2_divisao_por_paciente.csv")
    print(f"  Divisão por paciente: treino {divisao['treino_consultas']} | teste {divisao['teste_consultas']} "
          f"| pacientes em ambos: {divisao['pacientes_em_ambos']}")

    # 5. Modelagem --------------------------------------------------------------
    print("\n5. Modelagem (busca em grade com validação cruzada agrupada)")
    print("  Experimento A - sem SMS_received")
    metricas_a, modelos_a, cv_a = modelagem.treinar_experimento("A", treino, teste, ATRIBUTOS_A)
    print("  Experimento B - com SMS_received")
    metricas_b, modelos_b, cv_b = modelagem.treinar_experimento("B", treino, teste, ATRIBUTOS_B)

    gravar_csv(modelagem.tabela_hiperparametros(cv_a), "tabela3_hiperparametros_A.csv")
    gravar_csv(modelagem.tabela_hiperparametros(cv_b), "hiperparametros_B.csv")
    gravar_csv(metricas_a, "tabela6_metricas_experimento_A.csv")
    gravar_csv(metricas_b, "tabela8_metricas_experimento_B.csv")

    # 6. Avaliação --------------------------------------------------------------
    print("\n6. Avaliação no conjunto de teste")
    print(metricas_a[modelagem.COLUNAS_METRICAS].round(3).to_string())

    gravar_figura(graficos.grafico_metricas(metricas_a), "figura4_metricas_modelos.png")
    gravar_figura(graficos.grafico_roc(modelos_a, teste, ATRIBUTOS_A), "figura5_curvas_roc.png")
    gravar_figura(graficos.grafico_matrizes_confusao(metricas_a), "apoio_matrizes_confusao.png")

    interpretacao = avaliacao.interpretar_modelos(modelos_a)
    regras = avaliacao.regras_da_arvore(modelos_a)
    gravar_csv(interpretacao, "interpretacao_modelos_A.csv")
    (PASTA_RESULTADOS / "regras_arvore_A.txt").write_text(regras, encoding="utf-8")

    print("\n  Bootstrap pareado por paciente (1.000 reamostragens)...")
    bootstrap = avaliacao.bootstrap_por_paciente(modelos_a, teste, ATRIBUTOS_A, metricas_a)
    gravar_csv(bootstrap, "tabela7_bootstrap.csv", index=False)
    print(bootstrap.round(4).to_string(index=False))

    print("\n  Análise de sensibilidade (divisão por linhas x por paciente)...")
    sensibilidade = avaliacao.sensibilidade_por_linhas(df, modelos_a, idx_treino_linhas, idx_teste_linhas,
                                                       ATRIBUTOS_A, metricas_a)
    gravar_csv(sensibilidade, "sensibilidade_divisao.csv")
    print(sensibilidade.round(3).to_string())

    # 7. Exportação -------------------------------------------------------------
    saida = {
        "arquivo": {"nome": ARQUIVO_CSV.name, "sha256": dados.calcular_sha256(), "bytes": ARQUIVO_CSV.stat().st_size},
        "base_original": resumo,
        "qualidade": qualidade.to_dict(orient="records"),
        "removidos": invalidos[["AppointmentID", "Age", "WaitingDays", "Motivo"]].to_dict(orient="records"),
        "base_limpa": {"consultas": len(df), "pacientes": int(df["PatientId"].nunique()),
                       "faltas": int(df["NoShow"].sum()), "taxa_falta": float(taxa_geral)},
        "taxa_por_espera": {str(k): {"consultas": int(v["consultas"]), "faltas": int(v["faltas"]),
                                     "taxa": float(v["taxa"])} for k, v in por_espera.iterrows()},
        "taxa_por_idade": {str(k): {"consultas": int(v["consultas"]), "faltas": int(v["faltas"]),
                                    "taxa": float(v["taxa"])} for k, v in por_idade.iterrows()},
        "sms": sms,
        "associacoes": assoc.to_dict(orient="records"),
        "vazamento_divisao_por_linhas": vazamento,
        "divisao_por_paciente": divisao,
        "experimento_A": {"atributos": ATRIBUTOS_A, "teste": metricas_para_dict(metricas_a), "validacao": cv_a},
        "experimento_B": {"atributos": ATRIBUTOS_B, "teste": metricas_para_dict(metricas_b), "validacao": cv_b},
        "interpretacao_A": interpretacao.round(4).to_dict(orient="index"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "sensibilidade": sensibilidade.to_dict(orient="index"),
        "versoes": {"python": platform.python_version(), "pandas": pd.__version__,
                    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                    "seaborn": seaborn.__version__},
    }
    (PASTA_RESULTADOS / "resultados.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2, default=para_json), encoding="utf-8")

    print(f"\nConcluído. Figuras em {PASTA_FIGURAS} e resultados em {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
