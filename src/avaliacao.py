"""Interpretação dos modelos, bootstrap por paciente e análise de sensibilidade."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import export_text

from .config import SEED
from .modelagem import MODELOS, calcular_metricas, pesos_balanceados, probabilidades


def interpretar_modelos(modelos):
    """Coeficientes da Regressão Logística e importância dos atributos na Árvore.

    Os coeficientes dos atributos numéricos estão na escala padronizada, então
    dá para comparar o tamanho entre eles. Os binários ficam na escala 0/1.
    """
    rl = modelos["Regressão Logística"]
    arvore = modelos["Árvore de Decisão"]
    coeficientes = pd.Series(rl.named_steps["clf"].coef_[0],
                             index=rl.named_steps["prep"].get_feature_names_out())
    importancias = pd.Series(arvore.named_steps["clf"].feature_importances_,
                             index=arvore.named_steps["prep"].get_feature_names_out())
    tabela = pd.DataFrame({"Coeficiente RL (padronizado)": coeficientes,
                           "Importância na árvore": importancias})
    return tabela.sort_values("Importância na árvore", ascending=False)


def regras_da_arvore(modelos):
    arvore = modelos["Árvore de Decisão"]
    nomes = list(arvore.named_steps["prep"].get_feature_names_out())
    return export_text(arvore.named_steps["clf"], feature_names=nomes, show_weights=False, decimals=1)


def _linhas_por_paciente(pacientes):
    """Para cada paciente do teste, as posições das suas consultas."""
    _, codigo = np.unique(pacientes, return_inverse=True)
    ordem = np.argsort(codigo, kind="stable")
    cortes = np.cumsum(np.bincount(codigo))[:-1]
    return np.split(ordem, cortes)


def bootstrap_por_paciente(modelos, teste, atributos, metricas_teste, n=1000, seed=SEED):
    """Bootstrap pareado: as diferenças entre modelos são maiores que o acaso? (Tabela 7)

    Em cada repetição, os pacientes do teste são sorteados com reposição e
    entram com todas as suas consultas. Os três modelos são avaliados na mesma
    amostra, por isso o bootstrap é pareado. Se o intervalo de 95% da
    diferença contém zero, a diferença não é conclusiva.
    """
    rng = np.random.default_rng(seed)
    y = teste["NoShow"].values
    prob = probabilidades({m: modelos[m] for m in MODELOS}, teste, atributos)
    grupos = _linhas_por_paciente(teste["PatientId"].values)
    n_pacientes = len(grupos)

    pares = [("MLP", "Árvore de Decisão"), ("MLP", "Regressão Logística"),
             ("Árvore de Decisão", "Regressão Logística")]
    diferencas = {par: {"F1-score": [], "ROC-AUC": []} for par in pares}

    for _ in range(n):
        sorteio = rng.integers(0, n_pacientes, n_pacientes)
        linhas = np.concatenate([grupos[i] for i in sorteio])
        y_amostra = y[linhas]
        f1 = {m: f1_score(y_amostra, prob[m][linhas] >= 0.5) for m in MODELOS}
        auc = {m: roc_auc_score(y_amostra, prob[m][linhas]) for m in MODELOS}
        for a, b in pares:
            diferencas[(a, b)]["F1-score"].append(f1[a] - f1[b])
            diferencas[(a, b)]["ROC-AUC"].append(auc[a] - auc[b])

    linhas_tabela = []
    for (a, b), por_metrica in diferencas.items():
        for metrica, valores in por_metrica.items():
            inferior, superior = np.percentile(valores, [2.5, 97.5])
            linhas_tabela.append({
                "Comparação": f"{a} − {b}",
                "Métrica": metrica,
                "Diferença": metricas_teste.loc[a, metrica] - metricas_teste.loc[b, metrica],
                "IC 95% inferior": inferior,
                "IC 95% superior": superior,
                "Conclusiva": bool(inferior > 0 or superior < 0),
            })
    return pd.DataFrame(linhas_tabela)


def sensibilidade_por_linhas(df, modelos, idx_treino, idx_teste, atributos, metricas_paciente):
    """Compara a divisão por linhas com a divisão por paciente.

    Os modelos são retreinados com os mesmos hiperparâmetros do Experimento A,
    agora na divisão aleatória por linhas, em que o mesmo paciente pode estar
    no treino e no teste. Essa análise não é usada para escolher o modelo.
    """
    X_treino, X_teste = df.loc[idx_treino, atributos], df.loc[idx_teste, atributos]
    y_treino, y_teste = df.loc[idx_treino, "NoShow"].values, df.loc[idx_teste, "NoShow"].values

    linhas = {}
    for nome in MODELOS:
        modelo = clone(modelos[nome])
        parametros_fit = {"clf__sample_weight": pesos_balanceados(y_treino)} if nome == "MLP" else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(X_treino, y_treino, **parametros_fit)
        prob = modelo.predict_proba(X_teste)[:, 1]
        metricas = calcular_metricas(y_teste, prob, (prob >= 0.5).astype(int))
        linhas[nome] = {
            "F1 (linhas)": metricas["F1-score"],
            "F1 (paciente)": metricas_paciente.loc[nome, "F1-score"],
            "ROC-AUC (linhas)": metricas["ROC-AUC"],
            "ROC-AUC (paciente)": metricas_paciente.loc[nome, "ROC-AUC"],
        }
    return pd.DataFrame(linhas).T
