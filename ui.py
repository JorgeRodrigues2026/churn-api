"""Componentes visuais reutilizáveis do dashboard (CSS, gráficos e rótulos)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- paleta ----------
PRIMARY = "#4F46E5"
SECONDARY = "#94A3B8"
RISK_COLORS = {"baixo": "#16A34A", "medio": "#F59E0B", "alto": "#DC2626"}
RISK_LABELS = {"baixo": "Baixo", "medio": "Médio", "alto": "Alto"}

# ---------- rótulos PT-BR para os valores aceitos pela API ----------
LABELS = {
    "Yes": "Sim", "No": "Não", "Male": "Masculino", "Female": "Feminino",
    "No internet service": "Sem internet", "No phone service": "Sem telefonia",
    "DSL": "DSL", "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mensal", "One year": "Anual", "Two year": "Bienal",
    "Electronic check": "Boleto eletrônico", "Mailed check": "Cheque pelo correio",
    "Bank transfer (automatic)": "Débito automático",
    "Credit card (automatic)": "Cartão de crédito (automático)",
}


def label(value: str) -> str:
    return LABELS.get(value, value)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.35); border-radius: 12px;
            padding: 14px 18px; background: rgba(148,163,184,.06);
        }
        div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
        .risk-badge {
            display: inline-block; padding: 6px 16px; border-radius: 999px;
            font-weight: 700; color: white; font-size: 1rem; letter-spacing: .3px;
        }
        .status-dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;}
        .app-subtitle {opacity:.7; margin-top:-.6rem; margin-bottom:1.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f"<div class='app-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def risk_badge(risk: str) -> str:
    return (f"<span class='risk-badge' style='background:{RISK_COLORS[risk]}'>"
            f"Risco {RISK_LABELS[risk]}</span>")


def _base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.2)")
    return fig


# ---------- gráficos ----------
def gauge(probability: float) -> go.Figure:
    pct = probability * 100
    color = RISK_COLORS["baixo" if pct < 30 else "medio" if pct < 60 else "alto"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probabilidade de churn"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "rgba(22,163,74,.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,.15)"},
                {"range": [60, 100], "color": "rgba(220,38,38,.15)"},
            ],
            "threshold": {"line": {"color": "#334155", "width": 3}, "value": 50},
        },
    ))
    return _base_layout(fig, 300)


def confusion_matrix_chart(cm: list[list[int]]) -> go.Figure:
    labels_x = ["Previsto: Fica", "Previsto: Churn"]
    labels_y = ["Real: Fica", "Real: Churn"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels_x, y=labels_y, colorscale=[[0, "#EEF2FF"], [1, PRIMARY]],
        showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 20},
        hovertemplate="%{y} · %{x}<br>%{z} clientes<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Matriz de confusão (teste)")
    return _base_layout(fig)


def model_comparison_chart(val_metrics: dict, selected: str) -> go.Figure:
    names = {"logistic_regression": "Regressão Logística", "random_forest": "Random Forest"}
    metric_names = {"accuracy": "Acurácia", "precision": "Precisão", "recall": "Recall",
                    "f1_score": "F1", "roc_auc": "ROC-AUC"}
    fig = go.Figure()
    for model, metrics in val_metrics.items():
        fig.add_bar(
            name=names.get(model, model) + (" ✓" if model == selected else ""),
            x=[metric_names[m] for m in metric_names], y=[metrics[m] for m in metric_names],
            marker_color=PRIMARY if model == selected else SECONDARY,
            text=[f"{metrics[m]:.2f}" for m in metric_names], textposition="outside",
        )
    fig.update_layout(title="Comparação na validação", barmode="group", yaxis_range=[0, 1.05])
    return _base_layout(fig)


def histogram(probs: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=probs * 100, nbinsx=20, marker_color=PRIMARY,
                                 hovertemplate="%{x}%<br>%{y} clientes<extra></extra>"))
    fig.add_vline(x=50, line_dash="dash", line_color="#334155",
                  annotation_text="limiar 50%", annotation_position="top")
    fig.update_layout(title="Distribuição das probabilidades",
                      xaxis_title="Probabilidade de churn (%)", yaxis_title="Clientes", bargap=0.05)
    return _base_layout(fig)


def risk_donut(counts: pd.Series) -> go.Figure:
    order = [r for r in ["baixo", "medio", "alto"] if r in counts.index]
    fig = go.Figure(go.Pie(
        labels=[RISK_LABELS[r] for r in order], values=[counts[r] for r in order], hole=.6,
        marker_colors=[RISK_COLORS[r] for r in order], sort=False,
        hovertemplate="%{label}: %{value} clientes (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Clientes por faixa de risco")
    return _base_layout(fig)


def bar_by_group(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    grouped = (df.groupby(group_col)["churn_probability"].mean().mul(100)
               .sort_values(ascending=False))
    fig = go.Figure(go.Bar(
        x=[label(v) for v in grouped.index], y=grouped.values, marker_color=PRIMARY,
        text=[f"{v:.1f}%" for v in grouped.values], textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title="Risco médio (%)",
                      yaxis_range=[0, max(grouped.max() * 1.25, 10)])
    return _base_layout(fig)


def whatif_line(x: list, probs: list[float], x_title: str, current_x) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=x, y=[p * 100 for p in probs], mode="lines+markers", line=dict(color=PRIMARY, width=3),
        hovertemplate=f"{x_title}: %{{x}}<br>Churn: %{{y:.1f}}%<extra></extra>",
    ))
    fig.add_hrect(y0=60, y1=100, fillcolor=RISK_COLORS["alto"], opacity=.07, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor=RISK_COLORS["medio"], opacity=.07, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=RISK_COLORS["baixo"], opacity=.07, line_width=0)
    fig.add_vline(x=current_x, line_dash="dot", line_color="#334155",
                  annotation_text="cliente atual")
    fig.update_layout(xaxis_title=x_title, yaxis_title="Probabilidade de churn (%)",
                      yaxis_range=[0, 100])
    return _base_layout(fig)


def whatif_bars(categories: list[str], probs: list[float], current: str) -> go.Figure:
    colors = [RISK_COLORS["baixo" if p < .3 else "medio" if p < .6 else "alto"] for p in probs]
    fig = go.Figure(go.Bar(
        x=[label(c) + (" (atual)" if c == current else "") for c in categories],
        y=[p * 100 for p in probs], marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs], textposition="outside",
    ))
    fig.update_layout(yaxis_title="Probabilidade de churn (%)", yaxis_range=[0, 105])
    return _base_layout(fig)
