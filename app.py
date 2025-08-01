# -*- coding: utf-8 -*-
import warnings, re
warnings.filterwarnings("ignore")

from itertools import cycle
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

# ----------------------------- 기본 설정 -----------------------------
st.set_page_config(page_title="배터리 시세 대시보드", layout="wide")
st.title("배터리/제품 월별 시세 분석 & 예측")

# ====== 1) CSV 로드 ======
uploaded = st.sidebar.file_uploader(
    "CSV 업로드(계약일·배터리종류·제품구분·개당가격 포함)", type=["csv"]
)
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/통합거래내역.csv")

# 🔸 1-A. 컬럼 이름 양쪽 공백 제거
df.columns = df.columns.str.strip()

# 🔸 1-B. 날짜 컬럼을 datetime으로 변환
df["계약일"] = pd.to_datetime(df["계약일"], errors="coerce")
df = df.dropna(subset=["계약일"])            # 날짜 변환 실패 행 제거

# 🔸 1-C. 개당가격을 순수 숫자로 변환(콤마·원 단위 제거)
df["개당가격"] = (
    df["개당가격"]
      .astype(str)
      .str.replace(r"[^\d.\-]", "", regex=True)   # 숫자·마이너스·점만 남김
      .replace("", pd.NA)
      .astype(float)
)

# ====== 2) 사이드바 설정 ======
col_type = st.sidebar.radio("분류 컬럼 선택", ["배터리종류", "제품구분"])
forecast_horizon = st.sidebar.number_input("예측 개월 수", 6, 36, 12)

palette = cycle(
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
)

# ====== 3) 통계·예측 함수 ======
@st.cache_data
def make_stats_and_forecast(df, category_col, horizon):
    groups = sorted(df[category_col].dropna().unique())
    color_map = {g: next(palette) for g in groups}
    monthly_stats_dict, forecast_dict, future_idx_dict = {}, {}, {}

    for g in groups:
        dfg = df[df[category_col] == g].copy()
        if dfg.empty:
            continue

        monthly_stats = (
            dfg.set_index("계약일")["개당가격"]
               .resample("M")
               .agg(["min", "max", "mean"])
               .dropna()
        )
        monthly_stats_dict[g] = monthly_stats

        if len(monthly_stats) >= 24:
            model = ExponentialSmoothing(
                monthly_stats["mean"],
                trend="add",
                seasonal="add",
                seasonal_periods=12,
            )
            fit = model.fit()
            future_idx = pd.date_range(
                start=monthly_stats.index[-1] + pd.offsets.MonthBegin(1),
                periods=horizon,
                freq="M",
            )
            forecast_dict[g] = fit.forecast(horizon)
            future_idx_dict[g] = future_idx
    return groups, color_map, monthly_stats_dict, forecast_dict, future_idx_dict

groups, color_map, monthly_stats_dict, forecast_dict, future_idx_dict = make_stats_and_forecast(
    df, col_type, forecast_horizon
)

# ====== 4) 그래프 1: 전체 비교 ======
fig1 = go.Figure()
for g, stats in monthly_stats_dict.items():
    fig1.add_trace(
        go.Scatter(
            x=stats.index,
            y=stats["mean"],
            mode="lines+markers",
            name=f"{g} 월평균",
            line=dict(color=color_map[g]),
        )
    )
    if g in forecast_dict:
        fig1.add_trace(
            go.Scatter(
                x=future_idx_dict[g],
                y=forecast_dict[g],
                mode="lines+markers",
                name=f"{g} 예측",
                line=dict(color=color_map[g], dash="dash"),
            )
        )

fig1.update_layout(
    title=f"{col_type}별 월평균 시세 & {forecast_horizon}개월 예측",
    xaxis_title="기간",
    yaxis_title="가격",
    hovermode="x unified",
)
fig1.update_yaxes(tickformat=",")
st.plotly_chart(fig1, use_container_width=True)

# ====== 5) 그래프 2: 에러바 & 드롭다운 ======
if groups:
    selected = st.selectbox(f"{col_type} 선택", groups, index=0)
    stats = monthly_stats_dict[selected]

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
            marker=dict(color=color_map[selected], size=8),
            name=selected,
        )
    )
    fig2.update_layout(
        title=f"{selected} 월별 시세 범위 (최저·평균·최고)",
        xaxis_title="기간",
        yaxis_title="가격",
    )
    fig2.update_yaxes(tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning(f"`{col_type}` 컬럼에 데이터가 없습니다.")
