"""Divisão treino/teste, modelos e métricas.

Pontos principais do desenho experimental:
- a divisão é agrupada por PatientId, para que nenhum paciente apareça ao
  mesmo tempo no treino e no teste;
- os hiperparâmetros são escolhidos por validação cruzada agrupada (5 dobras)
  usando só o treino, com o F1-score da classe falta como critério;
- PatientId e AppointmentID nunca entram como atributos dos modelos;
- o limiar de decisão é 0,5.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import NUMERICOS, SEED

MODELOS = ["Regressão Logística", "Árvore de Decisão", "MLP"]
REFERENCIA = "Referência (classe majoritária)"
COLUNAS_METRICAS = ["Acurácia", "Precisão", "Recall", "F1-score", "ROC-AUC"]
COLUNAS_MATRIZ = ["VN", "FP", "FN", "VP"]


# ---------------------------------------------------------------------------
# Divisão dos dados
# ---------------------------------------------------------------------------

def dividir_por_linhas(df):
    """Divisão aleatória comum, 80/20 e estratificada.

    Não é usada para avaliar os modelos. Serve para mostrar o problema do
    vazamento entre pacientes e para a análise de sensibilidade.
    """
    return train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=df["NoShow"])


def medir_vazamento(df, idx_treino, idx_teste):
    """Conta quantas consultas do teste são de pacientes que também estão no treino."""
    pacientes_treino = set(df.loc[idx_treino, "PatientId"])
    pacientes_teste = df.loc[idx_teste, "PatientId"]
    compartilhadas = pacientes_teste.isin(pacientes_treino)
    return {
        "consultas_teste": int(len(idx_teste)),
        "pacientes_teste": int(pacientes_teste.nunique()),
        "pacientes_em_ambos": int(len(pacientes_treino & set(pacientes_teste))),
        "consultas_teste_com_paciente_no_treino": int(compartilhadas.sum()),
        "proporcao": float(compartilhadas.mean()),
    }


def dividir_por_paciente(df):
    """Divisão usada na avaliação final (Tabela 2 do artigo).

    StratifiedGroupKFold com 5 dobras: a primeira dobra (cerca de 20%) vira o
    teste e as outras quatro formam o treino. Todos os registros de um mesmo
    paciente ficam do mesmo lado.
    """
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    idx_treino, idx_teste = next(divisor.split(df, df["NoShow"], groups=df["PatientId"]))
    return df.iloc[idx_treino], df.iloc[idx_teste]


def resumo_divisao(treino, teste):
    total = len(treino) + len(teste)
    return {
        "treino_consultas": int(len(treino)),
        "teste_consultas": int(len(teste)),
        "treino_pacientes": int(treino["PatientId"].nunique()),
        "teste_pacientes": int(teste["PatientId"].nunique()),
        "pacientes_em_ambos": len(set(treino["PatientId"]) & set(teste["PatientId"])),
        "taxa_falta_treino": float(treino["NoShow"].mean()),
        "taxa_falta_teste": float(teste["NoShow"].mean()),
        "proporcao_teste": len(teste) / total,
    }


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def preparador(atributos, padronizar):
    """Padroniza os atributos numéricos (se pedido) e repassa os binários sem mudança."""
    numericos = [c for c in NUMERICOS if c in atributos]
    transformacao = StandardScaler() if padronizar else "passthrough"
    return ColumnTransformer([("num", transformacao, numericos)],
                             remainder="passthrough", verbose_feature_names_out=False)


def montar_modelos(atributos):
    """Para cada modelo: pipeline, grade de hiperparâmetros e se usa peso por amostra.

    O desbalanceamento (cerca de 20% de faltas) é tratado com pesos.
    A Regressão Logística e a Árvore usam class_weight='balanced'. O
    MLPClassifier não tem class_weight, então recebe pesos por amostra
    equivalentes. A árvore não precisa de padronização.
    """
    return {
        "Regressão Logística": (
            Pipeline([
                ("prep", preparador(atributos, padronizar=True)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},
            False,
        ),
        "Árvore de Decisão": (
            Pipeline([
                ("prep", preparador(atributos, padronizar=False)),
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
            ]),
            {"clf__max_depth": [3, 5, 7, 9, 12], "clf__min_samples_leaf": [1, 50, 200]},
            False,
        ),
        "MLP": (
            Pipeline([
                ("prep", preparador(atributos, padronizar=True)),
                ("clf", MLPClassifier(early_stopping=True, max_iter=300, random_state=SEED)),
            ]),
            {"clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32)], "clf__alpha": [1e-4, 1e-3, 1e-2]},
            True,
        ),
    }


def calcular_metricas(y, prob, pred):
    """Métricas da classe falta (1) e a matriz de confusão."""
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Acurácia": accuracy_score(y, pred),
        "Precisão": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1-score": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, prob),
        "VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
    }


def treinar_experimento(nome, treino, teste, atributos, verbose=True):
    """Treina a referência e os três modelos e avalia todos no conjunto de teste.

    Retorna a tabela de métricas no teste, os modelos já ajustados e as
    informações da validação cruzada (melhores parâmetros e F1 médio).
    """
    X_treino, X_teste = treino[atributos], teste[atributos]
    y_treino, y_teste = treino["NoShow"].values, teste["NoShow"].values
    grupos = treino["PatientId"].values
    pesos = compute_sample_weight("balanced", y_treino)
    validacao = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    resultados, modelos, info_cv = {}, {}, {}

    referencia = DummyClassifier(strategy="most_frequent").fit(X_treino, y_treino)
    resultados[REFERENCIA] = calcular_metricas(
        y_teste, referencia.predict_proba(X_teste)[:, 1], referencia.predict(X_teste))

    for modelo, (pipeline, grade, usa_peso) in montar_modelos(atributos).items():
        inicio = time.time()
        busca = GridSearchCV(pipeline, grade, scoring="f1", cv=validacao, n_jobs=-1, refit=True)
        parametros_fit = {"clf__sample_weight": pesos} if usa_peso else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            busca.fit(X_treino, y_treino, groups=grupos, **parametros_fit)

        melhor = busca.best_estimator_
        prob = melhor.predict_proba(X_teste)[:, 1]
        resultados[modelo] = calcular_metricas(y_teste, prob, (prob >= 0.5).astype(int))
        modelos[modelo] = melhor

        i = busca.best_index_
        info_cv[modelo] = {
            "melhores_parametros": {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v)
                                    for k, v in busca.best_params_.items()},
            "f1_cv_media": float(busca.cv_results_["mean_test_score"][i]),
            "f1_cv_dp": float(busca.cv_results_["std_test_score"][i]),
            "segundos": round(time.time() - inicio, 1),
        }
        if verbose:
            print(f"  [{nome}] {modelo}: {info_cv[modelo]['melhores_parametros']} "
                  f"| F1 médio na validação = {info_cv[modelo]['f1_cv_media']:.3f} "
                  f"({info_cv[modelo]['segundos']} s)")

    metricas = pd.DataFrame(resultados).T[COLUNAS_METRICAS + COLUNAS_MATRIZ]
    metricas[COLUNAS_MATRIZ] = metricas[COLUNAS_MATRIZ].astype(int)
    metricas[COLUNAS_METRICAS] = metricas[COLUNAS_METRICAS].astype(float)
    return metricas, modelos, info_cv


def tabela_hiperparametros(info_cv):
    """Resumo da busca em grade no formato da Tabela 3 do artigo."""
    linhas = []
    for modelo, info in info_cv.items():
        parametros = "; ".join(f"{k} = {v}" for k, v in info["melhores_parametros"].items())
        linhas.append({"Modelo": modelo, "Configuração selecionada": parametros,
                       "F1 médio na validação": info["f1_cv_media"],
                       "Desvio padrão": info["f1_cv_dp"]})
    return pd.DataFrame(linhas).set_index("Modelo")


def probabilidades(modelos, dados, atributos):
    return {nome: modelo.predict_proba(dados[atributos])[:, 1] for nome, modelo in modelos.items()}


def pesos_balanceados(y):
    return compute_sample_weight("balanced", np.asarray(y))
