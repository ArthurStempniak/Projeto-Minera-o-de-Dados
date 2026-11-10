# Base de dados

Arquivo usado: `KaggleV2-May-2016.csv`, do conjunto *Medical Appointment No Shows*,
publicado por Joni Hoppen no Kaggle:
https://www.kaggle.com/datasets/joniarroba/noshowappointments

A base tem 110.527 consultas de 62.299 pacientes atendidos em unidades públicas
de saúde de Vitória (ES), com consultas entre 29/04/2016 e 08/06/2016.

## Colunas

| Coluna | Descrição |
|---|---|
| PatientId | Identificador do paciente (usado só para agrupar a divisão treino/teste) |
| AppointmentID | Identificador da consulta (não usado nos modelos) |
| Gender | Sexo registrado (F/M) |
| ScheduledDay | Data e hora em que a consulta foi agendada |
| AppointmentDay | Data da consulta (sempre com hora 00:00) |
| Age | Idade em anos |
| Neighbourhood | Bairro da unidade (81 bairros de Vitória; não usado) |
| Scholarship | Participação no Bolsa Família (0/1) |
| Hipertension | Hipertensão registrada (0/1) |
| Diabetes | Diabetes registrada (0/1) |
| Alcoholism | Alcoolismo registrado (0/1) |
| Handcap | Código de deficiência (0 a 4; os códigos 2 a 4 não são documentados) |
| SMS_received | Se o paciente recebeu SMS de lembrete (0/1) |
| No-show | `Yes` = faltou, `No` = compareceu |

Os nomes das colunas estão como no arquivo original, inclusive as grafias
`Hipertension` e `Handcap`.
