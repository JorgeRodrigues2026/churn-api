"""
Churn Insights — Dashboard Streamlit que consome a Churn Prediction API (FastAPI).

Executar (com a API já rodando em outro terminal):
    uvicorn app.main:app --reload
    streamlit run dashboard/app.py

Arquitetura: Streamlit (interface) -> api_client.py (HTTP/REST) -> FastAPI -> modelo.
O dashboard não importa o modelo nem o pacote `app`: conversa só por HTTP.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from api_client import DEFAULT_API_URL, ApiResult, ChurnApiClient

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT_DIR / "examples" / "clientes_amostra.csv"

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
ADDONS = {
    "OnlineSecurity": "Segurança online", "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de dispositivo", "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV", "StreamingMovies": "Streaming de filmes",
}
DEFAULT_PROFILE = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 191.4,
}

st.set_page_config(page_title="Churn Insights", page_icon="📉", layout="wide")
ui.inject_css()


# =====================================================================
# Sidebar: conexão com a API e navegação
# =====================================================================
with st.sidebar:
    st.markdown("### 📉 Churn Insights")
    st.caption("Painel de retenção de clientes")
    api_url = st.text_input("URL da API", value=st.session_state.get("api_url", DEFAULT_API_URL))
    st.session_state["api_url"] = api_url
    client = ChurnApiClient(api_url)

    health = client.health()
    online = health.ok and health.data.get("model_loaded")
    color = "#16A34A" if online else ("#F59E0B" if health.ok else "#DC2626")
    status_txt = "Online · modelo carregado" if online else (
        "API online · modelo indisponível" if health.ok else "API offline")
    st.markdown(f"<span class='status-dot' style='background:{color}'></span>{status_txt}",
                unsafe_allow_html=True)
    if health.ok:
        st.caption(f"Latência: {health.elapsed_ms} ms · API v{health.data.get('api_version')}")

    st.divider()
    page = st.radio(
        "Navegação",
        ["Visão geral", "Predição individual", "Simulador what-if",
         "Análise em lote", "Laboratório da API"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"[Swagger /docs]({api_url}/docs) · [ReDoc]({api_url}/redoc)")
    st.caption("Projeto Avaliativo · SCTEC · Módulo 2")

if not health.ok:
    st.error(f"**Sem conexão com a API.** {health.error}")
    st.code("uvicorn app.main:app --reload", language="bash")
    st.info("Suba a API em outro terminal e recarregue esta página.")
    st.stop()


# =====================================================================
# Helpers
# =====================================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_model_info(url: str) -> dict | None:
    res = ChurnApiClient(url).model_info()
    return res.data if res.ok else None


def show_api_error(res: ApiResult) -> None:
    st.error(f"**HTTP {res.status_code or '—'}** · {res.error}")


def retention_actions(c: dict) -> list[str]:
    """Sugestões heurísticas de retenção baseadas no perfil (não vêm do modelo)."""
    actions = []
    if c["Contract"] == "Month-to-month":
        actions.append("📄 Oferecer migração para plano **anual** com desconto.")
    if c["PaymentMethod"] == "Electronic check":
        actions.append("💳 Incentivar **pagamento automático** (débito ou cartão).")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        actions.append("🛠️ Incluir **suporte técnico** gratuito por 3 meses.")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        actions.append("🔒 Oferecer **segurança online** como bônus de fidelidade.")
    if c["tenure"] < 12:
        actions.append("🤝 Acionar programa de **onboarding** para clientes novos.")
    if c["MonthlyCharges"] > 80:
        actions.append("💰 Revisar o plano: mensalidade acima da média da base.")
    return actions or ["✅ Perfil estável: manter relacionamento e pesquisa de satisfação."]


def prepare_batch(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """Converte um CSV no formato da base Telco em payloads válidos para a API."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return [], missing
    data = df[FEATURES].copy()
    if pd.api.types.is_numeric_dtype(data["SeniorCitizen"]):
        data["SeniorCitizen"] = data["SeniorCitizen"].map({1: "Yes", 0: "No"})
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    data["tenure"] = data["tenure"].astype(int)
    return data.to_dict(orient="records"), []


def curl_for(payload: dict, path: str = "/v1/predict") -> str:
    return (f"curl -X POST {api_url}{path} \\\n  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(payload, ensure_ascii=False)}'")


# =====================================================================
# Página 1 — Visão geral
# =====================================================================
def page_overview() -> None:
    ui.page_header("Visão geral do modelo",
                   "Saúde do serviço e desempenho do modelo em produção, lidos de "
                   "<code>GET /health</code> e <code>GET /v1/model/info</code>.")

    h = health.data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status da API", "Online" if h["status"] == "ok" else "Degradada")
    c2.metric("Modelo em RAM", "Sim" if h["model_loaded"] else "Não")
    c3.metric("Algoritmo", {"logistic_regression": "Reg. Logística",
                           "random_forest": "Random Forest"}.get(h.get("model_name"), "—"))
    c4.metric("Versão do modelo", h.get("model_version") or "—")

    info = load_model_info(api_url)
    if not info:
        st.warning("Métricas indisponíveis (models/metrics.json não encontrado na API).")
        return

    st.markdown("#### Desempenho no conjunto de teste")
    st.caption(f"Base: {info['dataset']} · {info['n_samples']:,} clientes · "
               f"treino {info['split']['train']:,} / validação {info['split']['validation']:,} "
               f"/ teste {info['split']['test']:,}".replace(",", "."))
    t = info["test_metrics"]
    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{t['roc_auc']:.3f}", help="Capacidade de ordenar clientes por risco")
    m[1].metric("Recall", f"{t['recall']:.1%}", help="% dos churns reais que o modelo encontra")
    m[2].metric("Precisão", f"{t['precision']:.1%}", help="% dos alertas que são churn de fato")
    m[3].metric("F1-Score", f"{t['f1_score']:.3f}")
    m[4].metric("Acurácia", f"{t['accuracy']:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ui.confusion_matrix_chart(info["test_confusion_matrix"]), width="stretch")
    with col2:
        st.plotly_chart(ui.model_comparison_chart(info["validation_metrics"],
                                                  info["selected_model"]), width="stretch")

    (tn, fp), (fn, tp) = info["test_confusion_matrix"]
    st.info(
        f"**Leitura de negócio:** dos {tp + fn} clientes que realmente cancelaram no teste, o "
        f"modelo identificou **{tp}** ({tp / (tp + fn):.0%}). Em troca, {fp} clientes fiéis "
        "receberiam uma ação de retenção desnecessária — custo aceitável frente à perda de um "
        "cliente.")


# =====================================================================
# Página 2 — Predição individual
# =====================================================================
def customer_form(base: dict) -> dict:
    """Formulário reativo que só permite combinações aceitas pelas regras de negócio da API."""
    def sel(col, text, key, options, **kw):
        return col.selectbox(text, options, index=options.index(base[key]) if base[key] in options
                             else 0, format_func=ui.label, key=f"f_{key}", **kw)

    st.markdown("##### 👤 Perfil")
    c = st.columns(4)
    gender = sel(c[0], "Gênero", "gender", ["Female", "Male"])
    senior = sel(c[1], "Idoso (65+)", "SeniorCitizen", ["No", "Yes"])
    partner = sel(c[2], "Possui parceiro(a)", "Partner", ["Yes", "No"])
    dependents = sel(c[3], "Possui dependentes", "Dependents", ["No", "Yes"])

    st.markdown("##### 📡 Serviços")
    c = st.columns(3)
    phone = sel(c[0], "Telefonia", "PhoneService", ["Yes", "No"])
    if phone == "Yes":
        lines = c[1].selectbox("Múltiplas linhas", ["No", "Yes"], format_func=ui.label,
                               index=1 if base["MultipleLines"] == "Yes" else 0, key="f_lines_on")
    else:
        lines = c[1].selectbox("Múltiplas linhas", ["No phone service"], format_func=ui.label,
                               disabled=True, key="f_lines_off")
    internet = sel(c[2], "Internet", "InternetService", ["Fiber optic", "DSL", "No"])

    addons = {}
    c = st.columns(3)
    for i, (key, text) in enumerate(ADDONS.items()):
        col = c[i % 3]
        if internet == "No":
            addons[key] = col.selectbox(text, ["No internet service"], format_func=ui.label,
                                        disabled=True, key=f"f_{key}_off")
        else:
            addons[key] = col.selectbox(text, ["No", "Yes"], format_func=ui.label,
                                        index=1 if base[key] == "Yes" else 0, key=f"f_{key}_on")

    st.markdown("##### 📄 Contrato e faturamento")
    c = st.columns(3)
    contract = sel(c[0], "Contrato", "Contract", ["Month-to-month", "One year", "Two year"])
    payment = sel(c[1], "Forma de pagamento", "PaymentMethod",
                  ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                   "Credit card (automatic)"])
    paperless = sel(c[2], "Fatura digital", "PaperlessBilling", ["Yes", "No"])

    c = st.columns(3)
    tenure = c[0].slider("Meses como cliente", 0, 72, int(base["tenure"]), key="f_tenure")
    monthly = c[1].number_input("Mensalidade (R$)", 1.0, 1000.0, float(base["MonthlyCharges"]),
                                step=5.0, key="f_monthly")
    auto_total = c[2].toggle("Total automático", value=True, key="f_auto")
    total = round(monthly * tenure, 2) if auto_total else c[2].number_input(
        "Total faturado (R$)", 0.0, 100000.0, float(base["TotalCharges"]), step=50.0, key="f_total")
    if auto_total:
        c[2].caption(f"Total faturado: **R$ {total:,.2f}**")

    return {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, **addons, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total),
    }


def page_single() -> None:
    ui.page_header("Predição individual",
                   "Preencha o perfil e envie para <code>POST /v1/predict</code>.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            customer = customer_form(st.session_state.get("profile", DEFAULT_PROFILE))
            submitted = st.button("🔮 Calcular risco de churn", type="primary", width="stretch")

    if submitted:
        with st.spinner("Consultando a API..."):
            st.session_state["single_result"] = client.predict(customer)
        st.session_state["profile"] = customer

    with right:
        res: ApiResult | None = st.session_state.get("single_result")
        if res is None:
            st.info("👈 Ajuste o perfil e clique em **Calcular risco de churn**.")
            return
        if not res.ok:
            show_api_error(res)
            return

        r = res.data
        with st.container(border=True):
            st.plotly_chart(ui.gauge(r["churn_probability"]), width="stretch")
            st.markdown(f"<div style='text-align:center'>{ui.risk_badge(r['risk_level'])}</div>",
                        unsafe_allow_html=True)
            verdict = "deve <b>cancelar</b>" if r["churn_prediction"] else "deve <b>permanecer</b>"
            st.markdown(f"<p style='text-align:center;margin-top:.8rem'>O modelo indica que o "
                        f"cliente {verdict}.</p>", unsafe_allow_html=True)
            st.caption(f"HTTP {res.status_code} · {res.elapsed_ms} ms · modelo v{r['model_version']}")

        st.markdown("##### Ações de retenção sugeridas")
        for action in retention_actions(res.request_payload):
            st.markdown(f"- {action}")
        st.caption("Sugestões por regras de negócio, complementares à predição do modelo.")

        with st.expander("🔌 Requisição e resposta REST"):
            st.markdown("**Request** `POST /v1/predict`")
            st.json(res.request_payload, expanded=False)
            st.markdown("**Response** `200 OK`")
            st.json(r)
            st.markdown("**cURL**")
            st.code(curl_for(res.request_payload), language="bash")


# =====================================================================
# Página 3 — Simulador what-if
# =====================================================================
def page_whatif() -> None:
    ui.page_header("Simulador what-if",
                   "Veja como o risco muda ao alterar uma característica do cliente. "
                   "Cada cenário é enviado em uma única chamada a <code>POST /v1/predict/batch</code>.")

    base = st.session_state.get("profile", DEFAULT_PROFILE)
    if "profile" not in st.session_state:
        st.caption("Usando um perfil de exemplo. Faça uma predição individual para simular o seu.")

    dimension = st.segmented_control(
        "Variável a simular",
        ["Tempo de casa", "Tipo de contrato", "Forma de pagamento", "Mensalidade"],
        default="Tempo de casa",
    ) or "Tempo de casa"

    def scenario(**changes) -> dict:
        s = {**base, **changes}
        s["TotalCharges"] = round(s["MonthlyCharges"] * s["tenure"], 2)
        return s

    if dimension == "Tempo de casa":
        xs = list(range(0, 73, 3))
        scenarios = [scenario(tenure=x) for x in xs]
    elif dimension == "Mensalidade":
        xs = list(range(20, 125, 5))
        scenarios = [scenario(MonthlyCharges=float(x)) for x in xs]
    elif dimension == "Tipo de contrato":
        xs = ["Month-to-month", "One year", "Two year"]
        scenarios = [scenario(Contract=x) for x in xs]
    else:
        xs = ["Electronic check", "Mailed check", "Bank transfer (automatic)",
              "Credit card (automatic)"]
        scenarios = [scenario(PaymentMethod=x) for x in xs]

    res = client.predict_batch(scenarios)
    if not res.ok:
        show_api_error(res)
        return
    probs = [p["churn_probability"] for p in res.data["predictions"]]

    with st.container(border=True):
        if dimension == "Tempo de casa":
            fig = ui.whatif_line(xs, probs, "Meses como cliente", base["tenure"])
        elif dimension == "Mensalidade":
            fig = ui.whatif_line(xs, probs, "Mensalidade (R$)", base["MonthlyCharges"])
        else:
            key = "Contract" if dimension == "Tipo de contrato" else "PaymentMethod"
            fig = ui.whatif_bars(xs, probs, base[key])
        st.plotly_chart(fig, width="stretch")

    best = min(range(len(probs)), key=probs.__getitem__)
    worst = max(range(len(probs)), key=probs.__getitem__)
    c = st.columns(3)
    c[0].metric("Menor risco", f"{probs[best]:.1%}", help=str(ui.label(str(xs[best]))))
    c[1].metric("Maior risco", f"{probs[worst]:.1%}", help=str(ui.label(str(xs[worst]))))
    c[2].metric("Amplitude", f"{(probs[worst] - probs[best]) * 100:.1f} p.p.")
    st.caption(f"{len(scenarios)} cenários em 1 requisição · {res.elapsed_ms} ms")


# =====================================================================
# Página 4 — Análise em lote
# =====================================================================
def page_batch() -> None:
    ui.page_header("Análise em lote",
                   "Envie um CSV no formato da base Telco. O painel divide em lotes de 100 "
                   "(limite da API) e consolida os resultados.")

    c1, c2 = st.columns([3, 1])
    uploaded = c1.file_uploader("Arquivo CSV", type="csv", label_visibility="collapsed")
    use_sample = c2.button("Usar amostra (300 clientes)", width="stretch",
                           disabled=not SAMPLE_CSV.exists())

    if uploaded is not None:
        st.session_state["batch_df"] = pd.read_csv(uploaded)
        st.session_state["batch_name"] = uploaded.name
    elif use_sample:
        st.session_state["batch_df"] = pd.read_csv(SAMPLE_CSV)
        st.session_state["batch_name"] = SAMPLE_CSV.name

    df_in: pd.DataFrame | None = st.session_state.get("batch_df")
    if df_in is None:
        st.info("Envie um CSV ou clique em **Usar amostra**. Colunas obrigatórias: "
                + ", ".join(f"`{c}`" for c in FEATURES))
        return

    records, missing = prepare_batch(df_in)
    if missing:
        st.error("Colunas ausentes no CSV: " + ", ".join(missing))
        return

    cache_key = (st.session_state["batch_name"], len(records), api_url)
    if st.session_state.get("batch_key") != cache_key:
        bar = st.progress(0.0, text=f"Enviando {len(records)} clientes para a API...")
        res = client.predict_many(records, progress_cb=lambda p: bar.progress(p))
        bar.empty()
        if not res.ok:
            show_api_error(res)
            return
        st.session_state["batch_res"] = res
        st.session_state["batch_key"] = cache_key
    res: ApiResult = st.session_state["batch_res"]

    preds = pd.DataFrame(res.data["predictions"])
    df = pd.concat([df_in.reset_index(drop=True), preds], axis=1)

    st.success(f"**{len(df)} clientes** analisados em {res.elapsed_ms / 1000:.2f} s "
               f"({-(-len(df) // 100)} requisições) · {st.session_state['batch_name']}")

    k = st.columns(4)
    k[0].metric("Clientes", f"{len(df)}")
    k[1].metric("Previstos para churn",
                f"{df['churn_prediction'].sum()} ({df['churn_prediction'].mean():.0%})")
    k[2].metric("Risco médio", f"{df['churn_probability'].mean():.1%}")
    k[3].metric("Risco alto", f"{(df['risk_level'] == 'alto').sum()}")

    if "Churn" in df.columns:
        real = df["Churn"].astype(str).str.strip().isin(["Yes", "1", "True"])
        pred = df["churn_prediction"]
        tp = int((real & pred).sum())
        recall = tp / max(real.sum(), 1)
        precision = tp / max(pred.sum(), 1)
        st.caption(f"📌 O CSV traz o rótulo real (`Churn`): acurácia **{(real == pred).mean():.1%}** · "
                   f"recall **{recall:.1%}** · precisão **{precision:.1%}** nesta amostra.")

    col1, col2 = st.columns([3, 2])
    col1.plotly_chart(ui.histogram(df["churn_probability"]), width="stretch")
    col2.plotly_chart(ui.risk_donut(df["risk_level"].value_counts()), width="stretch")

    col1, col2 = st.columns(2)
    col1.plotly_chart(ui.bar_by_group(df, "Contract", "Risco médio por contrato"), width="stretch")
    col2.plotly_chart(ui.bar_by_group(df, "PaymentMethod", "Risco médio por pagamento"),
                      width="stretch")

    st.markdown("#### 🎯 Clientes prioritários para retenção")
    f1, f2 = st.columns([2, 1])
    risks = f1.multiselect("Faixas de risco", ["alto", "medio", "baixo"], default=["alto"],
                           format_func=lambda r: ui.RISK_LABELS[r])
    contracts = f2.multiselect("Contrato", sorted(df["Contract"].unique()), format_func=ui.label)
    view = df[df["risk_level"].isin(risks)]
    if contracts:
        view = view[view["Contract"].isin(contracts)]
    view = view.sort_values("churn_probability", ascending=False)

    cols = [c for c in ["customerID", "churn_probability", "risk_level", "Contract", "tenure",
                        "MonthlyCharges", "PaymentMethod", "InternetService", "Churn"]
            if c in view.columns]
    st.dataframe(
        view[cols], width="stretch", hide_index=True, height=380,
        column_config={
            "customerID": "Cliente",
            "churn_probability": st.column_config.ProgressColumn(
                "Prob. churn", min_value=0, max_value=1, format="percent"),
            "risk_level": "Risco", "Contract": "Contrato", "tenure": "Meses",
            "MonthlyCharges": st.column_config.NumberColumn("Mensalidade", format="R$ %.2f"),
            "PaymentMethod": "Pagamento", "InternetService": "Internet", "Churn": "Churn real",
        },
    )
    st.download_button("⬇️ Baixar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="predicoes_churn.csv", mime="text/csv")


# =====================================================================
# Página 5 — Laboratório da API
# =====================================================================
LAB_SCENARIOS = [
    ("Cliente válido", "POST", "/v1/predict", DEFAULT_PROFILE, 200),
    ("Texto em campo numérico (tenure = 'dez')", "POST", "/v1/predict",
     {**DEFAULT_PROFILE, "tenure": "dez"}, 422),
    ("Valor fora do limite (tenure = -5)", "POST", "/v1/predict",
     {**DEFAULT_PROFILE, "tenure": -5}, 422),
    ("Categoria inexistente (Contract = 'Weekly')", "POST", "/v1/predict",
     {**DEFAULT_PROFILE, "Contract": "Weekly"}, 422),
    ("Campo obrigatório ausente (sem MonthlyCharges)", "POST", "/v1/predict",
     {k: v for k, v in DEFAULT_PROFILE.items() if k != "MonthlyCharges"}, 422),
    ("Campo extra não permitido", "POST", "/v1/predict", {**DEFAULT_PROFILE, "hacker": 1}, 422),
    ("Regra de negócio: sem internet, com streaming", "POST", "/v1/predict",
     {**DEFAULT_PROFILE, "InternetService": "No"}, 400),
    ("Lote vazio", "POST", "/v1/predict/batch", {"customers": []}, 422),
    ("Health check", "GET", "/health", None, 200),
]


def page_lab() -> None:
    ui.page_header("Laboratório da API",
                   "Dispara requisições válidas e inválidas para comprovar a validação "
                   "do Pydantic (422), as regras de negócio (400) e os status semânticos.")

    if st.button("▶️ Executar todos os cenários", type="primary"):
        rows = []
        for name, method, path, payload, expected in LAB_SCENARIOS:
            r = client._request(method, path, payload)
            rows.append({"Cenário": name, "Rota": f"{method} {path}", "Esperado": expected,
                         "Recebido": r.status_code, "OK": "✅" if r.status_code == expected else "❌",
                         "ms": r.elapsed_ms, "Resposta": r.error or "sucesso",
                         "_payload": payload})
        st.session_state["lab_rows"] = rows

    rows = st.session_state.get("lab_rows")
    if not rows:
        st.info("Clique para executar os 9 cenários contra a API.")
        return

    passed = sum(r["OK"] == "✅" for r in rows)
    c = st.columns(3)
    c[0].metric("Cenários", len(rows))
    c[1].metric("Aprovados", passed)
    c[2].metric("Latência média", f"{sum(r['ms'] for r in rows) / len(rows):.1f} ms")

    table = pd.DataFrame(rows).drop(columns="_payload")
    st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("Ver payload de um cenário"):
        idx = st.selectbox("Cenário", range(len(rows)), format_func=lambda i: rows[i]["Cenário"])
        st.json(rows[idx]["_payload"] if rows[idx]["_payload"] is not None else {})


PAGES = {
    "Visão geral": page_overview, "Predição individual": page_single,
    "Simulador what-if": page_whatif, "Análise em lote": page_batch,
    "Laboratório da API": page_lab,
}
PAGES[page]()
