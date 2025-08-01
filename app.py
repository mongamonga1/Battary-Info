# -*- coding: utf-8 -*-
"""
Home · Main page
왼쪽 사이드바 메뉴(pages/…)에서 각 세부 분석을 실행할 수 있도록
간단한 개요·통계·미리보기를 보여주는 대시보드 역할을 합니다.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# ───────────────────── 페이지 기본 설정 ─────────────────────
st.set_page_config(
    page_title="배터리 데이터 분석 허브",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔋 배터리/제품 통합 분석 대시보드")

# ───────────────────── 데이터 로드 ─────────────────────
DATA_PATH = Path("data/통합거래내역.csv")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # 컬럼 공백 제거 & 날짜·가격 정제
    df.columns = df.columns.str.strip()
    df["계약일"] = pd.to_datetime(df["계약일"], errors="coerce")
    df["개당가격"] = (
        df["개당가격"]
          .astype(str)
          .str.replace(r"[^\d.\-]", "", regex=True)
          .pipe(pd.to_numeric, errors="coerce")
    )
    return df.dropna(subset=["계약일"])

df = load_data(DATA_PATH)

# ───────────────────── 사이드바 안내 ─────────────────────
with st.sidebar:
    st.header("🗂 메뉴")
    st.write("왼쪽 **사이드바 상단**의 페이지 목록에서")
    st.write("• *car kmeans*  \n• *recommend system*  \n• *forest lstm*  \n• *timeseries analysis*")
    st.write("각 분석 페이지를 선택해 보세요!")
    st.divider()
    # (선택) 사용자 CSV 업로드 → 임시로 미리보기
    up = st.file_uploader("CSV 업로드(미리보기용)", type="csv")
    if up:
        tmp_df = pd.read_csv(up, nrows=100)
        st.success(f"업로드 파일 미리보기 (100행)")
        st.dataframe(tmp_df, use_container_width=True)

# ───────────────────── 데이터가 있을 때 대시보드 출력 ─────────────────────
if df is None:
    st.warning("`data/통합거래내역.csv` 파일이 없거나 찾을 수 없습니다. "
               "레포의 **data/** 폴더에 CSV를 올려 두면 요약 통계를 볼 수 있어요.")
    st.stop()

# 요약 지표
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 거래 건수", f"{len(df):,}")
col2.metric("판매업체 수", df["판매업체"].nunique())
col3.metric("구매업체 수", df["구매업체"].nunique())
col4.metric(
    "관측 기간",
    f"{df['계약일'].min().date()} ↔ {df['계약일'].max().date()}",
)

st.divider()

# 월별 거래 건수 추이
st.subheader("📊 월별 거래 건수")
monthly_cnt = (
    df.set_index("계약일")
      .resample("M")["계약번호"]
      .count()
      .rename("count")
)
st.line_chart(monthly_cnt, use_container_width=True)

# 상위 10개 배터리종류 거래량
if "배터리종류" in df.columns:
    st.subheader("🔝 상위 10개 배터리종류 거래 건수")
    top_batt = (
        df["배터리종류"]
          .value_counts()
          .head(10)
          .sort_values(ascending=True)
    )
    st.bar_chart(top_batt)

st.divider()
st.subheader("데이터 미리보기 (앞 50행)")
st.dataframe(df.head(50), use_container_width=True)

st.caption("© 2025 Battery-Info ― 메뉴에서 상세 분석을 확인하세요.")
