# -*- coding: utf-8 -*-
"""📈 Timeseries analysis ― 배터리·제품 월별 시세 & 예측"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from itertools import cycle
from pathlib import Path
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ───────────────────── 페이지 설정 ─────────────────────
st.header("📈 시계열 분석 (월평균 & Holt-Winters 예측)")

# ───────────────────── 1) 데이터 로드 ─────────────────────
DATA_PATH = Path("data/통합거래내역.csv")
up_file   = st.sidebar.file_uploader("CSV 업로드(계약일·배터리종류·제품구분·개당가격 포함)", type="csv")

if up_file:
    df = pd.read_csv(up_file)
    st.success("업로드한 CSV를 사용합니다.")
elif DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    st.error("`data/통합거래내역.csv` 파일을 찾을 수 없습니다.\n사이드바에서 CSV를 업로드해 주세요.")
    st.stop()

# ── 컬럼 정리 ──
df.columns = df.columns.str.strip()
df["계약일"]   = pd.to_datetime(df["계약일"], errors="coerce")
df["개당가격"] = (
    df["개당가격"]
      .astype(str)
      .str.replace(r"[^\d.\-]", "", regex=True)
      .pipe(pd.to_numeric, errors="coerce")
)
df = df.dropna(subset=["계약일"])

# ───────────────────── 2) 사이드바 옵션 ─────────────────────
col_type = st.sidebar.radio("분류 기준", ["배터리종류", "제품구분"])
forecast_horizon = st.sidebar.slider("예측 개월 수", 6, 36, 12)
sel_items = st.sidebar.multiselect(
    f"{col_type} 선택 (최대 5개)", 
    sorted(df[col_type].dropna().unique()),
    default=None,
    max_selections=5
)

if not sel_items:
    st.info("사이드바에서 분석할 항목을 선택하세요.")
    st.stop()

palette = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
color_map = {k: next(palette) for k in sel_items}

# ───────────────────── 3) 통계·예측 계산 ─────────────────────
monthly_stats, forecasts, future_idx = {}, {}, {}

for item in sel_items:
    dfg = df[df[col_type] == item].copy()
    grp = (
        dfg.set_index("계약일")["개당가격"]
           .resample("M")
           .agg(["min", "max", "mean"])
           .dropna()
    )
    if grp.empty:
        continue
    monthly_stats[item] = grp

    if len(grp) >= 24:
        model = ExponentialSmoothing(
            grp["mean"], trend="add", seasonal="add", seasonal_periods=12
        ).fit()
        idx = pd.date_range(
            start=grp.index[-1] + pd.offsets.MonthBegin(1),
            periods=forecast_horizon,
            freq="M",
        )
        forecasts[item]   = model.forecast(forecast_horizon)
        future_idx[item]  = idx

# ───────────────────── 4) Plotly 그래프: 월평균 + 예측 ─────────────────────
fig = go.Figure()
for item, stats in monthly_stats.items():
    fig.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["mean"],
            mode="lines+markers",
            name=f"{item} 월평균",
            line=dict(color=color_map[item]),
        )
    )
    if item in forecasts:
        fig.add_trace(
            go.Scatter(
                x=future_idx[item],
                y=forecasts[item],
                mode="lines+markers",
                name=f"{item} 예측",
                line=dict(color=color_map[item], dash="dash"),
            )
        )

fig.update_layout(
    title=f"{col_type}별 월평균 시세 & {forecast_horizon}개월 예측 (범례 클릭 가능)",
    xaxis_title="기간",
    yaxis_title="가격",
    hovermode="x unified",
)
fig.update_yaxes(tickformat=",")
st.plotly_chart(fig, use_container_width=True)

# ───────────────────── 5) 선택 항목 상세(error-bar) ─────────────────────
sel_single = st.selectbox(f"상세 보기 ({col_type})", sel_items, index=0)
stats = monthly_stats.get(sel_single)
if stats is not None:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["mean"],
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=stats["max"] - stats["mean"],
                arrayminus=stats["mean"] - stats["min"],
                width=4,
                thickness=1.5,
            ),
            marker=dict(color=color_map[sel_single], size=8),
            name="평균 ± (최저↔최고)",
        )
    )
    fig2.update_layout(
        title=f"{sel_single} 월별 시세 범위 (최저·평균·최고)",
        xaxis_title="기간",
        yaxis_title="가격",
    )
    fig2.update_yaxes(tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)
