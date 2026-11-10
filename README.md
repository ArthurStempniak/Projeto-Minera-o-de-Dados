# MediConnect: previsão do não comparecimento a consultas médicas

Trabalho da disciplina de Mineração de Dados do curso de Engenharia de Software
do Centro Universitário Assis Gurgacz, Cascavel, 2026.

- **Aluno:** Arthur Stempniak
- **Professor:** Pedro Henrique De Avila Tonin

Este repositório tem o código da parte prática do artigo, que está em
[docs/MediConnect_Artigo_Mineracao_de_Dados.pdf](docs/MediConnect_Artigo_Mineracao_de_Dados.pdf).
A análise foi feita primeiro no Google Colab e depois organizada aqui em
módulos Python, para que possa ser lida e executada com mais facilidade.

## Sobre o trabalho

O MediConnect é uma proposta de plataforma de telemedicina e triagem remota.
Faltas em consultas deixam horários ociosos e atrapalham o planejamento das
agendas. A pergunta do trabalho é se as informações disponíveis no momento do
agendamento ajudam a identificar consultas com maior chance de falta, para
que a plataforma possa priorizar lembretes ou pedidos de confirmação.

Como o MediConnect ainda não tem dados próprios, foi usada a base pública
*Medical Appointment No Shows* (Hoppen, Kaggle), com 110.527 consultas de
Vitória (ES) em 2016. Os resultados servem como prova de conceito do método.

O trabalho segue as etapas do CRISP-DM. Alguns cuidados orientaram o desenho
experimental:

- a divisão entre treino e teste foi feita **por paciente**, para que a mesma
  pessoa nunca apareça nos dois conjuntos;
- os hiperparâmetros foram escolhidos por validação cruzada agrupada, usando
  só o conjunto de treino;
- o experimento principal (A) não usa o recebimento de SMS, porque essa
  informação só existe depois do agendamento. O experimento B inclui o SMS
  apenas para comparação.

## Estrutura do repositório

```
├── data/
│   ├── KaggleV2-May-2016.csv     base original, sem alterações
│   └── README.md                 origem, licença e descrição das colunas
├── docs/
│   └── MediConnect_Artigo_Mineracao_de_Dados.pdf
├── notebook/
│   └── analise_mediconnect.ipynb análise passo a passo, com os resultados comentados
├── src/
│   ├── config.py                 caminhos, semente e listas de atributos
│   ├── dados.py                  carga, verificação de qualidade e preparação
│   ├── exploracao.py             taxas de falta e testes de associação
│   ├── modelagem.py              divisão por paciente, modelos e métricas
│   ├── avaliacao.py              interpretação, bootstrap e sensibilidade
│   └── graficos.py               figuras do artigo e gráficos de apoio
├── figures/                      figuras geradas
├── results/                      tabelas (CSV), regras da árvore e resultados.json
├── executar_analise.py           roda o trabalho inteiro de uma vez
└── requirements.txt
```

## Como executar

É preciso ter o Python 3.13 (ou próximo) instalado. Na pasta do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux ou macOS
pip install -r requirements.txt
```

Para rodar tudo e gerar de novo as figuras e as tabelas:

```bash
python executar_analise.py
```

Para acompanhar a análise passo a passo:

```bash
jupyter notebook notebook/analise_mediconnect.ipynb
```

A execução completa leva entre 10 e 15 minutos num computador comum, quase
todo esse tempo na busca de hiperparâmetros do MLP. Os resultados do artigo foram obtidos
com as versões fixadas no `requirements.txt`. Com outras versões das
bibliotecas, o MLP pode apresentar diferenças pequenas na terceira casa decimal.

## Etapas do CRISP-DM no código

| Etapa | Onde está |
|---|---|
| Entendimento do negócio | seção 1 do notebook e introdução do artigo |
| Entendimento dos dados | `src/dados.py` (`resumo_base`, `verificar_qualidade`) |
| Preparação dos dados | `src/dados.py` (`registros_invalidos`, `preparar_dados`) |
| Análise exploratória | `src/exploracao.py` e `src/graficos.py` |
| Modelagem | `src/modelagem.py` (`dividir_por_paciente`, `treinar_experimento`) |
| Avaliação | `src/modelagem.py` (`calcular_metricas`) e `src/avaliacao.py` |
| Implantação | tratada só como proposta, na seção 6 do artigo |

## Relação com o artigo

| No artigo | Arquivo gerado |
|---|---|
| Tabela 1 – Verificações de qualidade | `results/tabela1_qualidade.csv` |
| Tabela 2 – Divisão agrupada por paciente | `results/tabela2_divisao_por_paciente.csv` |
| Tabela 3 – Hiperparâmetros do Experimento A | `results/tabela3_hiperparametros_A.csv` |
| Tabela 4 – Taxa de falta por tempo de espera | `results/tabela4_falta_por_espera.csv` |
| Tabela 5 – Associações com o não comparecimento | `results/tabela5_associacoes.csv` |
| Tabela 6 – Métricas no teste, Experimento A | `results/tabela6_metricas_experimento_A.csv` |
| Tabela 7 – Bootstrap pareado por paciente | `results/tabela7_bootstrap.csv` |
| Tabela 8 – Métricas do Experimento B | `results/tabela8_metricas_experimento_B.csv` |
| Figura 1 – Distribuição do desfecho | `figures/figura1_distribuicao_desfecho.png` |
| Figura 2 – Falta por tempo de espera | `figures/figura2_falta_por_espera.png` |
| Figura 3 – Falta por faixa etária | `figures/figura3_falta_por_idade.png` |
| Figura 4 – Métricas dos modelos | `figures/figura4_metricas_modelos.png` |
| Figura 5 – Curvas ROC | `figures/figura5_curvas_roc.png` |

Os arquivos com prefixo `apoio_` são gráficos e tabelas extras, usados na
análise mas não incluídos no texto do artigo. O arquivo `results/resultados.json`
reúne todos os números em um só lugar.

## Principais resultados

Depois de remover seis registros inválidos, ficaram 110.521 consultas, com
taxa geral de falta de 20,2%. O tempo de espera foi o sinal mais claro: a taxa
vai de 4,6% nas consultas do mesmo dia a 34,2% quando a espera é de 31 a 60
dias. Adolescentes e adultos jovens faltam mais que pessoas acima de 60 anos.
Os demais atributos têm associação fraca.

Métricas no conjunto de teste do Experimento A (sem SMS), com 22.105 consultas
de pacientes que não estavam no treino:

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Referência (classe majoritária) | 0,798 | 0,000 | 0,000 | 0,000 | 0,500 |
| Regressão Logística | 0,691 | 0,330 | 0,515 | 0,403 | 0,668 |
| Árvore de Decisão | 0,591 | 0,305 | 0,799 | 0,441 | 0,715 |
| MLP | 0,581 | 0,302 | 0,822 | 0,442 | 0,723 |

O MLP e a Árvore de Decisão ficaram empatados em F1 (o bootstrap não mostrou
diferença conclusiva). A precisão perto de 0,30 indica muitos falsos
positivos, então o modelo só faz sentido para ações de baixo custo, como um
lembrete adicional, e não para decisões que limitem o atendimento.

## Limitações

A base cobre cerca de seis semanas de 2016 em Vitória e não representa o
cenário atual de Cascavel nem do MediConnect. A divisão foi por paciente, mas
não temporal. O SMS não tem data de envio nem critério documentado. O limiar
de decisão ficou em 0,5 e as probabilidades não foram calibradas. Qualquer uso
com dados reais de pacientes exige análise jurídica e cuidados compatíveis
com a LGPD.

## Referências principais

- CHAPMAN, P. et al. *CRISP-DM 1.0: step-by-step data mining guide*. SPSS, 2000.
- HOPPEN, J. *Medical Appointment No Shows*. Kaggle, [2016?]. Disponível em:
  https://www.kaggle.com/datasets/joniarroba/noshowappointments.
- PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *Journal of
  Machine Learning Research*, v. 12, p. 2825–2830, 2011.

A lista completa de referências está no artigo.
