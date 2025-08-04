# -*- coding: utf-8 -*-
"""
Home · Main page (방법B 적용)
- 기본 Streamlit Pages 내비게이션/검색 숨김
- 사이드바에 커스텀 메뉴(st.page_link) 배치
- 시안형 대시보드 레이아웃
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ───────────────────── 페이지 기본 설정 ─────────────────────
st.set_page_config(
    page_title="배터리 데이터 분석 허브",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  /* 브랜드 텍스트(맨 위) */
  [data-testid="stSidebar"] .brand{
    font-weight: 900; font-size: 18px; letter-spacing: .6px;
    color: #ffffff; margin: 4px 0 12px 2px;
  }
  [data-testid="stSidebar"] .menu-title{
    color:#cfe0ff; margin: 6px 0 8px 0;
  }

  /* st.page_link로 생성된 링크의 텍스트를 '내부 요소까지' 밝게 강제 */
  [data-testid="stSidebar"] a[href]{
    color:#EAF2FF !important;        /* 링크 자체 색 */
    opacity:1 !important;
    display:block; padding:10px 12px; border-radius:10px; font-weight:700;
  }
  /* 앵커 내부의 p/span/div에도 동일 색/불투명도 상속 강제 */
  [data-testid="stSidebar"] a[href] *{
    color:inherit !important;
    opacity:1 !important;
    filter:none !important;
  }

  /* 호버/선택 상태는 배경만 살짝 강조 */
  [data-testid="stSidebar"] a[href]:hover{
    background:#13233b !important; color:#ffffff !important;
  }
  [data-testid="stSidebar"] a[aria-current="page"]{
    background:#1c2e4a !important; color:#ffffff !important;
    box-shadow: inset 0 0 0 1px #273b5c;
  }
  [data-testid="stSidebar"] a[aria-current="page"] *{
    color:inherit !important; opacity:1 !important;
  }
</style>
""", unsafe_allow_html=True)

# ───────────────────── 기본 Pages 내비/검색 숨기기 ─────────────────────
st.markdown(
    """
    <style>
      /* Streamlit 기본 Pages 내비게이션(검색 + 목록) 완전히 숨김 */
      [data-testid="stSidebarNav"] { display: none !important; }
      nav[aria-label="Pages"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── 공통 스타일(CSS) ─────────────────────
st.markdown(
    """
    <style>
      .app-container { background: #f6f8fb; }
      [data-testid="stAppViewContainer"] { background: #f6f8fb; }
      [data-testid="stHeader"] { background: rgba(246,248,251,0.7); backdrop-filter: blur(6px); }
      [data-testid="stSidebar"] { background: #0f1b2d; color: #d7e1f2; }
      [data-testid="stSidebar"] * { font-weight: 500; }

      /* 커스텀 메뉴(우리 손으로 만든 링크) */
      .menu-link {
        display:flex; align-items:center; gap:.5rem;
        padding:10px 12px; margin:4px 0; border-radius:10px;
        color:#e6efff; text-decoration:none; font-weight:600;
      }
      .menu-link:hover { background:#13233b; color:#fff; }
      .menu-section-title { color:#cfe0ff; font-weight:800; letter-spacing:.2px; }

      /* 카드 공통 */
      .kpi-card {
        border-radius: 14px; padding: 16px 18px; background: #fff;
        box-shadow: 0 2px 14px rgba(16,24,40,0.06); border: 1px solid #eef2f7; height: 100%;
      }
      .kpi-title { font-size: 13px; color:#7a8aa0; margin-bottom: 6px; display:flex; gap:8px; align-items:center;}
      .kpi-value { font-size: 26px; font-weight: 700; }
      .kpi-trend-up { color:#10b981; font-weight:700; }
      .kpi-trend-down { color:#ef4444; font-weight:700; }

      .box { background:#fff; border:1px solid #eef2f7; border-radius:14px; padding:14px; box-shadow:0 2px 14px rgba(16,24,40,.06); }
      .box-title { font-weight:700; color:#0f172a; display:flex; align-items:center; gap:10px; }
      .muted { color:#8a99ad; font-size:13px; }
      .blank { height:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── 타이틀/브레드크럼 ─────────────────────
today = pd.Timestamp.today()
week_of_month = (today.day - 1) // 7 + 1
st.markdown(
    f"""
    <div class="app-container">
      <h1 style="margin:0 0 6px 0;">🔋 배터리/제품 통합 분석 대시보드</h1>
      <div class="muted">Welcome  ·  <b>메인 화면</b>  ·  {today.strftime('%m월')} {week_of_month}주차</div>
      <div class="blank"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── 데이터 로드 ─────────────────────
DATA_PATH = Path("data/통합거래내역.csv")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "계약일" in df.columns:
        df["계약일"] = pd.to_datetime(df["계약일"], errors="coerce")
    if "개당가격" in df.columns:
        df["개당가격"] = (
            df["개당가격"].astype(str)
            .str.replace(r"[^\d.\-]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return df

df = load_data(DATA_PATH)

# ───────────────────── 사이드바: 커스텀 메뉴(방법B) ─────────────────────
with st.sidebar:
    st.markdown("### 📂 메뉴", help="상단 기본 Pages 네비 대신 커스텀 메뉴를 사용합니다.")
    # ⚠️ 실제 파일명으로 경로를 맞추세요. 예: 'pages/01_car kmeans.py'
    st.page_link("pages/2_car_kmeans.py",           label="군집 분석",          icon="🚗")
    st.page_link("pages/4_recommend_system.py",     label="기업 추천",    icon="✨")
    st.page_link("pages/5_forest_lstm.py",          label="이상거래 의심",         icon="🌳")
    st.page_link("pages/5_timeseries_analysis.py",  label="시세 분석", icon="📈")

# ───────────────────── 데이터 유무 방어 ─────────────────────
if df is None or ("계약일" not in df.columns):
    st.warning(
        "`data/통합거래내역.csv`가 없거나 **계약일** 컬럼이 없습니다. "
        "레포의 **data/** 폴더에 CSV를 두면 요약/차트가 채워집니다."
    )
    # 데모 데이터
    df = pd.DataFrame({
        "계약일": pd.date_range(end=today, periods=120, freq="D"),
        "계약번호": [f"T{i:05d}" for i in range(120)],
        "판매업체": np.random.choice(["A사","B사","C사","D사","E사"], 120),
        "구매업체": np.random.choice(["X사","Y사","Z사"], 120),
        "배터리종류": np.random.choice(["Kona","IONIQ5","EV6","GENESIS","PORTER2"], 120),
        "개당가격": np.random.randint(1200000, 2600000, 120)
    })

# ───────────────────── 상단 KPI 카드 ─────────────────────
total_cnt = len(df)
seller_n  = df["판매업체"].nunique() if "판매업체" in df.columns else 0
buyer_n   = df["구매업체"].nunique() if "구매업체" in df.columns else 0
period_txt = f"{pd.to_datetime(df['계약일']).min().date()} ↔ {pd.to_datetime(df['계약일']).max().date()}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">🧪 신규 <span class="muted">Battery</span></div>
          <div class="kpi-value">{total_cnt:,} 건</div>
          <div class="muted">지난달 대비 <span class="kpi-trend-down">-2</span></div>
        </div>
        """, unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">♻️ 재제조 및 재사용</div>
          <div class="kpi-value">{int(total_cnt*0.25):,} 건</div>
          <div class="muted">변동 <span class="kpi-trend-up">+3</span></div>
        </div>
        """, unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">🔁 재활용</div>
          <div class="kpi-value">{int(total_cnt*0.15):,} 건</div>
          <div class="muted">변동 <span class="kpi-trend-down">-5</span></div>
        </div>
        """, unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">📈 현황</div>
          <div class="kpi-value">{seller_n:,} / {buyer_n:,}</div>
          <div class="muted">관측 기간: {period_txt}</div>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown('<div class="blank"></div>', unsafe_allow_html=True)

# ───────────────────── 중앙: (좌) 라인차트  ·  (우) 이상거래 리스트 ─────────────────────
left, right = st.columns([4, 1.8])

with left:
    st.markdown('<div class="box"><div class="box-title">📉 시세 / 트렌드</div>', unsafe_allow_html=True)
    monthly_cnt = (
        pd.to_datetime(df["계약일"])
          .to_frame(name="계약일")
          .set_index("계약일")
          .resample("M")
          .size()
          .rename("count")
          .reset_index()
    )
    fig_line = px.line(monthly_cnt, x="계약일", y="count", markers=True)
    fig_line.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="box"><div class="box-title">🚨 이상거래 의심 내역</div>', unsafe_allow_html=True)
    if "개당가격" in df.columns:
        df2 = df.sort_values("계약일").copy()
        df2["변동"] = df2["개당가격"].pct_change().fillna(0)
        label_col = next((c for c in ["배터리종류", "모델", "차종", "판매업체"] if c in df2.columns), df2.columns[0])
        top_issue = (df2.tail(40)
                        .nlargest(6, "변동")
                        .assign(change=lambda d: (d["변동"]*100).round(2),
                                price=lambda d: d["개당가격"].map(lambda x: f"₩ {x:,.0f}")))
        low_issue = (df2.tail(40)
                        .nsmallest(6, "변동")
                        .assign(change=lambda d: (d["변동"]*100).round(2),
                                price=lambda d: d["개당가격"].map(lambda x: f"₩ {x:,.0f}")))
        issue = pd.concat([top_issue, low_issue]).head(9)
        for _, r in issue.iterrows():
            arrow = "🔺" if r["change"] >= 0 else "🔻"
            color = "#10b981" if r["change"] >= 0 else "#ef4444"
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;padding:8px 6px;border-bottom:1px solid #f0f3f7;">
                  <div style="font-weight:600;">{r[label_col]}</div>
                  <div style="font-variant-numeric: tabular-nums;">
                    <span style="margin-right:10px;color:#64748b;">{r['price']}</span>
                    <span style="color:{color};">{arrow} {abs(r['change']):.2f}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
    else:
        st.info("가격 컬럼이 없어 최근 거래 기준의 단순 목록만 표시합니다.")
        for s in df.head(9).index:
            st.markdown(f"- 항목 {s}")
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────── 하단: (좌) 고객 지원 · (우) 분석 결과 ─────────────────────
c_left, c_right = st.columns([2.4, 2])

with c_left:
    st.markdown('<div class="box"><div class="box-title">🧑‍💼 고객 지원</div>', unsafe_allow_html=True)
    demo_support = pd.DataFrame({
        "Date": [today.strftime("%Y/%m/%d %H:%M:%S"),
                 (today - pd.Timedelta("1D")).strftime("%Y/%m/%d %H:%M:%S"),
                 (today - pd.Timedelta("2D")).strftime("%Y/%m/%d %H:%M:%S")],
        "제목": ["이상거래 의심 제보", "이상거래 소명", "데이터 정합성 문의"],
        "사용자": ["이**(d****)", "김**(f******)", "박**(k*****)"],
    })
    st.dataframe(demo_support, use_container_width=True, height=240)
    st.markdown('</div>', unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="box"><div class="box-title">📌 Kona 주요 분석 결과</div>', unsafe_allow_html=True)
    metrics = ["안전성", "효율", "잔존수명", "온도안정", "전압균형"]
    radar_vals = np.clip(np.random.normal(loc=[70,65,68,72,66], scale=6), 40, 95)
    radar = go.Figure(
        data=[go.Scatterpolar(r=radar_vals.tolist()+[radar_vals[0]], theta=metrics+metrics[:1],
                              fill='toself', name="Kona")],
        layout=go.Layout(margin=dict(l=10,r=10,t=10,b=10), height=250,
                         polar=dict(radialaxis=dict(visible=True, range=[0,100])))
    )
    st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

    if "배터리종류" in df.columns:
        top_batt = df["배터리종류"].value_counts().head(6).reset_index()
        top_batt.columns = ["배터리종류", "count"]
        bar = px.bar(top_batt, x="배터리종류", y="count")
        bar.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=260)
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
    else:
        demo = pd.DataFrame({"배터리종류": list("ABCDEF"), "count": [9,7,6,5,4,3]})
        st.plotly_chart(px.bar(demo, x="배터리종류", y="count"), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────── 데이터 미리보기 ─────────────────────
st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
with st.expander("데이터 미리보기 (앞 50행)"):
    st.dataframe(df.head(50), use_container_width=True)

st.caption("© 2025 Battery-Info ― 사이드바 커스텀 메뉴에서 상세 분석 페이지로 이동하세요.")
