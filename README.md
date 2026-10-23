# 📉 Churn Prediction API

API REST em **FastAPI** que prevê se um cliente de telecom vai **cancelar o contrato (churn)**, servindo um pipeline de Machine Learning treinado com scikit-learn.

> Projeto Avaliativo — Módulo 2, Semana 13 · Desenvolvimento Análise Preditiva · SCTEC
> Autor: Jorge Luiz Susin Rodrigues · [GitHub](https://github.com/JorgeRodrigues2026)

---

## 1. Nome e resumo do projeto

Conquistar um cliente novo custa bem mais do que reter um atual. Esta API recebe os dados de um cliente e devolve a **probabilidade de cancelamento** e uma **faixa de risco** (baixo / médio / alto), permitindo ao time de retenção priorizar contatos e ofertas *antes* que o cliente saia.

## 2. Origem do modelo

**Opção A** — reaproveitamento e aprimoramento do projeto de **Análise de Churn / Regressão Logística** do curso.

- **Base:** IBM Telco Customer Churn — 7.043 clientes, 19 features, ~26,5% de churn ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). Arquivo em `data/telco_churn.csv`.
- **Aprimoramentos:** pipeline único (pré-processamento + modelo) serializado, divisão treino/validação/teste, comparação com Random Forest e tratamento do desbalanceamento com `class_weight="balanced"`.

### Pipeline (Fase 1)

| Etapa | Decisão | Justificativa |
|---|---|---|
| Limpeza | `TotalCharges` → numérico; 11 vazios viram `NaN` | Coluna vinha como texto |
| Imputação | Mediana (numéricas) / moda (categóricas) **dentro do pipeline** | Evita vazamento de dados do teste para o treino |
| Codificação | `OneHotEncoder(handle_unknown="ignore")` | Categoria nova em produção não derruba a API |
| Escalonamento | `StandardScaler` | Necessário para a Regressão Logística |
| Divisão | 60% treino / 20% validação / 20% teste, estratificada | Validação escolhe o modelo, teste mede só no final |
| Seleção | Maior ROC-AUC na validação | Mede a capacidade de **ordenar** clientes por risco |

### Métricas

| Modelo | Conjunto | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Regressão Logística | validação | 0,750 | 0,518 | 0,824 | 0,636 | **0,859** |
| Random Forest | validação | 0,759 | 0,530 | 0,816 | 0,643 | 0,859 |
| **Regressão Logística (escolhido)** | **teste** | **0,743** | **0,510** | **0,808** | **0,625** | **0,832** |

Matriz de confusão no teste: `[[745, 290], [72, 302]]` → o modelo encontra **81% dos clientes que realmente cancelam**.

**Por que a Regressão Logística?** Empatou com o Random Forest em ROC-AUC, mas é mais simples, mais rápida, mais leve (artefato de ~9 KB) e **interpretável** — dá para explicar ao negócio quais fatores aumentam o risco. O recall alto é proposital: para retenção, deixar de identificar um cliente que vai sair custa mais do que oferecer um desconto a quem ficaria.

Relatório completo em `models/metrics.json` (também exposto em `GET /v1/model/info`).

## 3. Tecnologias utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | 3.11+ | Linguagem |
| FastAPI | 0.143.2 | Framework da API |
| Uvicorn | 0.53.0 | Servidor ASGI |
| Pydantic | 2.13.5 | Validação de entrada e saída |
| scikit-learn | 1.9.1 | Pipeline de ML |
| pandas / numpy | 3.0.5 / 2.5.3 | Manipulação de dados |
| joblib | 1.6.0 | Serialização do modelo |
| pytest / httpx | 9.1.1 / 0.28.1 | Testes automatizados |
| Streamlit | 1.66.0 | Dashboard interativo |
| Plotly | 7.1.0 | Gráficos do dashboard |
| requests | 2.34.2 | Cliente REST do dashboard |

## 4. Instalação e execução

```bash
# 1. Clonar
git clone https://github.com/JorgeRodrigues2026/churn-api.git
cd churn-api

# 2. Ambiente virtual
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Dependências
pip install -r requirements.txt

# 4. (Opcional) Re-treinar o modelo — o artefato já vem em models/
python training/train_model.py

# 5. Subir a API
uvicorn app.main:app --reload # terminal 1: API

# 6. Rodar os testes
pytest -v

# 7. Dashboard (em OUTRO terminal, com a API rodando)       
streamlit run dashboard/app.py       # terminal 2: dashboard → http://localhost:8501
```

Acesse **http://127.0.0.1:8000/docs** (Swagger) e o dashboard em **http://localhost:8501**.

### Dashboard Streamlit — *Churn Insights*

Interface visual que consome a API **apenas via HTTP/REST** (`dashboard/api_client.py`), exatamente como um sistema externo faria — o dashboard não importa o modelo.

| Página | Endpoint usado | O que faz |
|---|---|---|
| Visão geral | `GET /health`, `GET /v1/model/info` | Status, KPIs do modelo, matriz de confusão, LogReg × RF |
| Predição individual | `POST /v1/predict` | Formulário reativo, velocímetro de risco, ações de retenção, request/response e cURL |
| Simulador what-if | `POST /v1/predict/batch` | Curva de risco por tempo de casa, contrato, pagamento ou mensalidade |
| Análise em lote | `POST /v1/predict/batch` | Upload de CSV (fatiado em lotes de 100), gráficos, ranking de prioridade, download |
| Laboratório da API | todos | Dispara 9 cenários e comprova 200 / 400 / 422 |

A URL da API é configurável na barra lateral ou pela variável `API_URL`.

## 5. Documentação da API

### Arquitetura em camadas (Fase 2)

```
churn-api/
├── app/
│   ├── main.py               # entrada, routers, lifespan, handler global de erros
│   ├── core/config.py        # caminhos relativos e configurações
│   ├── schemas/churn.py      # contratos Pydantic (entrada e saída)
│   ├── services/model_service.py  # carga do modelo, regras de negócio, inferência
│   └── routers/
│       ├── health.py         # GET /health
│       ├── predict.py        # POST /v1/predict e /v1/predict/batch
│       └── dependencies.py   # injeção do serviço que está na RAM
├── training/train_model.py   # Fase 1: treino e serialização
├── data/telco_churn.csv
├── models/churn_pipeline.joblib + metrics.json
├── dashboard/
│   ├── app.py                # interface Streamlit (5 páginas)
│   ├── api_client.py         # cliente REST da API
│   └── ui.py                 # gráficos Plotly e estilos
├── examples/                 # payloads JSON + clientes_amostra.csv
├── tests/test_api.py         # 13 testes
└── requirements.txt
```

**Fluxo de uma requisição:** `router` recebe → `Pydantic` valida (422) → `service` aplica regras de negócio (400) e chama o modelo já em memória → `router` devolve o schema de saída (200).

### Ciclo de vida (Fase 3)

O `.joblib` é carregado **uma única vez** no `lifespan` (`@asynccontextmanager`) e guardado em `app.state`; as rotas o recebem via `Depends`. Nenhuma rota lê o disco — isso é verificado pelo teste `test_model_loaded_once_not_per_request`, que faz 5 predições e confirma **1** chamada a `joblib.load`. Se o arquivo não existir, a API sobe em modo `degraded` em vez de cair.

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status do serviço e se o modelo está na RAM |
| POST | `/v1/predict` | Predição de um cliente |
| POST | `/v1/predict/batch` | Predição de 1 a 100 clientes |
| GET | `/v1/model/info` | Métricas do modelo em produção |

### Exemplo — `POST /v1/predict`

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @examples/cliente_alto_risco.json
```

Entrada:

```json
{
  "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
  "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 191.4
}
```

Saída (200):

```json
{ "churn_prediction": true, "churn_probability": 0.8865, "risk_level": "alto", "model_version": "1.0.0" }
```

### Status codes (Fase 5)

| Código | Quando | Exemplo |
|---|---|---|
| **200** | Sucesso | `examples/cliente_fiel.json` |
| **400** | Tipos válidos, mas combinação incoerente (regra de negócio) | `examples/regra_negocio_400.json` — sem internet, mas com streaming |
| **422** | Tipo errado, campo faltando/extra ou valor fora dos limites | `examples/invalido_422.json` — `"tenure": "dez"` |
| **500** | Falha inesperada na inferência (tratada, a API não cai) | — |
| **503** | Modelo não carregado na memória | — |

**Regras de validação (Fase 4):** `tenure` 0–120 · `MonthlyCharges` > 0 e ≤ 1000 · `TotalCharges` 0–100.000 · campos categóricos aceitam só os valores do `Literal` · campos desconhecidos são rejeitados (`extra="forbid"`) · lote de 1 a 100 clientes.

## 6. Links das evidências

- 🎥 Vídeo de apresentação: **[COLOCAR AQUI O LINK DO GOOGLE DRIVE]**
- 📉 Apresentação Git.io: https://jorgerodrigues2026.github.io/churn-api
- 💻 Repositório: https://github.com/JorgeRodrigues2026/churn-api

## Melhorias futuras

- Ajustar o limiar de decisão pelo custo real de retenção (hoje 0,5).
- Explicabilidade por cliente (coeficientes / SHAP) na resposta.
- Registro de versões com MLflow e monitoramento de *drift*.
- Dockerfile e deploy em nuvem.
