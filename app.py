# -*- coding: utf-8 -*-
"""
Battery-Info Streamlit App
──────────────────────────
• 홈 + 다중 페이지 네비게이션
• 3_OCR.py 페이지(📝) 사이드바 단독 추가
• Streamlit ≥ 1.34 필요
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ═════════════════════════ 기본 설정 ═════════════════════════
st.set_page_config(
    page_title="배터리 데이터 분석 허브",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
PAGES = {
    "kmeans":     ROOT / "pages/2_car_kmeans.py",
    "ocr":        ROOT / "3_OCR.py",                       # 🆕 OCR
    "reco":       ROOT / "pages/4_recommend_system.py",
    "fraud":      ROOT / "5_forest_lstm.py",
    "timeseries": ROOT / "pages/5_timeseries_analysis.py",
}

# 없는 파일 경고만
missing = [p for p in PAGES.values() if not p.exists()]
if missing:
    st.sidebar.warning(
        "다음 파일을 찾지 못했습니다:\n- " +
        "\n- ".join(str(m.relative_to(ROOT)) for m in missing)
    )

# ═════════════════════════ 홈 화면 ═════════════════════════
def render_home() -> None:
    # ---------- CSS ----------
    st.markdown(
        """
        <style>
          .app-container{background:#f6f8fb;}
          [data-testid="stAppViewContainer"]{background:#f6f8fb;}
          [data-testid="stHeader"]{background:rgba(246,248,251,.7);backdrop-filter:blur(6px);}
          [data-testid="stSidebar"]{background:#0f1b2d;color:#d7e1f2;}
          [data-testid="stSidebar"] *{font-weight:500;}
          [data-testid="stSidebar"] a[href]{
            color:#EAF2FF!important;opacity:1!important;display:block;padding:10px 12px;
            border-radius:10px;font-weight:700;
          }
          [data-testid="stSidebar"] a[href]:hover{
            background:#13233b!important;color:#fff!important;
          }
          [data-testid="stSidebar"] a[aria-current="page"]{
            background:#1c2e4a!important;color:#fff!important;box-shadow:inset 0 0 0 1px #273b5c;
          }
          .kpi-card{border-radius:14px;padding:16px 18px;background:#fff;
                    box-shadow:0 2px 14px rgba(16,24,40,.06);border:1px solid #eef2f7;height:100%;}
          .kpi-title{font-size:13px;color:#7a8aa0;margin-bottom:6px;display:flex;gap:8px;align-items:center;}
          .kpi-value{font-size:26px;font-weight:700;}
          .kpi-trend-up{color:#10b981;font-weight:700;}
          .kpi-trend-down{color:#ef4444;font-weight:700;}
          .box{background:#fff;border:1px solid #eef2f7;border-radius:14px;padding:14px;
               box-shadow:0 2px 14px rgba(16,24,40,.06);}
          .box-title{font-weight:700;color:#0f172a;display:flex;align-items:center;gap:10px;}
          .muted{color:#8a99ad;font-size:13px;}
          .blank{height:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 헤더 ----------
    today = pd.Timestamp.today()
    week = (today.day - 1) // 7 + 1
    st.markdown(
        f"""
        <div class='app-container'>
          <h1 style='margin:0 0 6px 0;'>🔋 배터리/제품 통합 분석 대시보드</h1>
          <div class='muted'>Welcome · <b>메인 화면</b> · {today.strftime('%m월')} {week}주차</div>
          <div class='blank'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 데이터 ----------
    DATA_PATH = ROOT / "data/통합거래내역.csv"

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
                df["개당가격"]
                .astype(str)
                .str.replace(r"[^\d.-]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
        return df

    df = load_data(DATA_PATH)
    if df is None or "계약일" not in df.columns:
        st.warning("`data/통합거래내역.csv` 를 찾지 못해 데모 데이터를 사용합니다.")
        df = pd.DataFrame(
            {
                "계약일": pd.date_range(end=today, periods=120, freq="D"),
                "계약번호": [f"T{i:05d}" for i in range(120)],
                "판매업체": np.random.choice(list("ABCDE"), 120),
                "구매업체": np.random.choice(list("XYZ"), 120),
                "배터리종류": np.random.choice(["Kona","IONIQ5","EV6","GENESIS","PORTER2"], 120),
                "개당가격": np.random.randint(1_200_000, 2_600_000, 120),
            }
        )

    # ---------- KPI 카드 ----------
    total_cnt = len(df)
    seller_n = df["판매업체"].nunique()
    buyer_n = df["구매업체"].nunique()
    period_txt = f"{df['계약일'].min().date()} ↔ {df['계약일'].max().date()}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-title'>🧪 신규 <span class='muted'>Battery</span></div>
              <div class='kpi-value'>{total_cnt:,} 건</div>
              <div class='muted'>지난달 대비 <span class='kpi-trend-down'>-2</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-title'>♻️ 재제조·재사용</div>
              <div class='kpi-value'>{int(total_cnt*0.25):,} 건</div>
              <div class='muted'>변동 <span class='kpi-trend-up'>+3</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-title'>🔁 재활용</div>
              <div class='kpi-value'>{int(total_cnt*0.15):,} 건</div>
              <div class='muted'>변동 <span class='kpi-trend-down'>-5</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-title'>📈 현황</div>
              <div class='kpi-value'>{seller_n:,} / {buyer_n:,}</div>
              <div class='muted'>관측 기간: {period_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)

    # ---------- 중앙 레이아웃 ----------
    left, right = st.columns([4, 1.8])

    # ▸ 좌: 월별 거래량
    with left:
        st.markdown('<div class="box"><div class="box-title">📉 시세 / 트렌드</div>', unsafe_allow_html=True)
        monthly_cnt = (
            pd.to_datetime(df["계약일"])
            .to_frame(name="계약일")
            .set_index("계약일")
            .resample("ME")
            .size()
            .rename("count")
            .reset_index()
        )
        fig_line = px.line(monthly_cnt, x="계약일", y="count", markers=True)
        fig_line.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ▸ 우: (데모) 최근 거래
    with right:
        st.markdown('<div class="box"><div class="box-title">🚨 이상거래 의심 내역</div>', unsafe_allow_html=True)
        for _, row in df.tail(6).iterrows():
            st.markdown(f"- {row['계약번호']} · ₩{row['개당가격']:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 꼬리말 ----------
    st.caption("© 2025 Battery-Info — 사이드바 메뉴로 상세 분석 페이지로 이동하세요.")

# ═════════════════════════ 페이지 등록 & 네비 ═════════════════════════
home      = st.Page(render_home,              title="🏠 홈", default=True, url_path="")
pg_kmeans = st.Page(str(PAGES["kmeans"]),     title="🚗 차명별 군집분석", url_path="kmeans")
pg_ocr    = st.Page(str(PAGES["ocr"]),        title="📝 OCR 판독",       url_path="ocr")
pg_reco   = st.Page(str(PAGES["reco"]),       title="✨ 기업 추천",      url_path="reco")
pg_fraud  = st.Page(str(PAGES["fraud"]),      title="🌳 이상거래 의심",  url_path="fraud")
pg_ts     = st.Page(str(PAGES["timeseries"]), title="📈 시세 분석",      url_path="timeseries")

current = st.navigation(
    [home, pg_kmeans, pg_ocr, pg_reco, pg_fraud, pg_ts],
    position="hidden",
)

# ═════════════════════════ 사이드바 ═════════════════════════
with st.sidebar:
    # 브랜드
    st.markdown(
        """
        <div style="position:sticky;top:0;z-index:10;background:#0f1b2d;
                    padding:12px 12px 6px;margin:0 -8px 8px -8px;
                    border-bottom:1px solid rgba(255,255,255,.06);">
          <div style="font-weight:900;font-size:24px;letter-spacing:.8px;color:#fff;">
            BATTERY-INFO
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.page_link(home, label="메인 화면", icon="🏠")
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # 그룹: 군집 - 추천 - 시세
    with st.expander("분석 결과 확인", expanded=True):
        st.page_link(pg_kmeans, label="군집 분석", icon="🚗")
        st.page_link(pg_reco,   label="기업 추천", icon="✨")
        st.page_link(pg_ts,     label="시세 분석", icon="📈")

    # 단독
    st.page_link(pg_fraud, label="이상거래 의심", icon="🌳")
    st.page_link(pg_ocr,   label="OCR 판독",   icon="📝")

# ═════════════════════════ 실행 ═════════════════════════
current.run()
