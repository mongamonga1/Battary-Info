# -*- coding: utf-8 -*-
"""
Main (Home) – Cloud-safe navigation
- 홈 화면을 함수로 감싸 st.Page(함수)로 등록
- st.navigation(...).run() 으로 현재 선택된 페이지 실행
- 사이드바 커스텀 메뉴(st.page_link) 유지
- pandas FutureWarning 대응 (resample("ME"), pct_change(fill_method=None))
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ───────────────────── 기본 설정 ─────────────────────
st.set_page_config(
    page_title="배터리 데이터 분석 허브",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
PAGES = {
    "kmeans": ROOT / "pages/2_car_kmeans.py",
    "ocr":    ROOT / "3_OCR.py",
    "reco": ROOT / "pages/4_recommend_system.py",
    "fraud": ROOT / "5_forest_lstm.py",
    "timeseries": ROOT / "pages/5_timeseries_analysis.py",
}

# 일부 페이지가 없더라도 앱은 계속 동작하게 경고만
missing = [p for p in PAGES.values() if not p.exists()]
if missing:
    st.sidebar.warning(
        "다음 페이지 파일을 찾지 못했습니다:\n- " + "\n- ".join(str(m.relative_to(ROOT)) for m in missing)
    )


# ───────────────────── 홈 화면 렌더러 ─────────────────────
def render_home():
    # ── 공통 스타일(CSS) ──
    st.markdown(
        """
        <style>
          .app-container { background: #f6f8fb; }
          [data-testid="stAppViewContainer"] { background: #f6f8fb; }
          [data-testid="stHeader"] { background: rgba(246,248,251,0.7); backdrop-filter: blur(6px); }
          [data-testid="stSidebar"] { background: #0f1b2d; color: #d7e1f2; }
          [data-testid="stSidebar"] * { font-weight: 500; }

          /* 사이드바 링크 스타일 */
          [data-testid="stSidebar"] a[href]{
            color:#EAF2FF !important; opacity:1 !important;
            display:block; padding:10px 12px; border-radius:10px; font-weight:700;
          }
          [data-testid="stSidebar"] a[href]:hover{ background:#13233b !important; color:#ffffff !important; }
          [data-testid="stSidebar"] a[aria-current="page"]{
            background:#1c2e4a !important; color:#ffffff !important; box-shadow: inset 0 0 0 1px #273b5c;
          }

          /* 카드/박스 공통 */
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

    # ── 타이틀/브레드크럼 ──
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

    # ── 데이터 로드 ──
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
                .str.replace(r"[^\d.\-]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )
        return df

    df = load_data(DATA_PATH)

    # ── 데이터 유무 방어 ──
    if df is None or ("계약일" not in df.columns):
        st.warning(
            "`data/통합거래내역.csv`가 없거나 **계약일** 컬럼이 없습니다. "
            "레포의 **data/** 폴더에 CSV를 두면 요약/차트가 채워집니다."
        )
        # 데모 데이터
        df = pd.DataFrame(
            {
                "계약일": pd.date_range(end=today, periods=120, freq="D"),
                "계약번호": [f"T{i:05d}" for i in range(120)],
                "판매업체": np.random.choice(["A사", "B사", "C사", "D사", "E사"], 120),
                "구매업체": np.random.choice(["X사", "Y사", "Z사"], 120),
                "배터리종류": np.random.choice(["Kona", "IONIQ5", "EV6", "GENESIS", "PORTER2"], 120),
                "개당가격": np.random.randint(1200000, 2600000, 120),
            }
        )

    # ── 상단 KPI 카드 ──
    total_cnt = len(df)
    seller_n = df["판매업체"].nunique() if "판매업체" in df.columns else 0
    buyer_n = df["구매업체"].nunique() if "구매업체" in df.columns else 0
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
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">♻️ 재제조 및 재사용</div>
              <div class="kpi-value">{int(total_cnt*0.25):,} 건</div>
              <div class="muted">변동 <span class="kpi-trend-up">+3</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">🔁 재활용</div>
              <div class="kpi-value">{int(total_cnt*0.15):,} 건</div>
              <div class="muted">변동 <span class="kpi-trend-down">-5</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">📈 현황</div>
              <div class="kpi-value">{seller_n:,} / {buyer_n:,}</div>
              <div class="muted">관측 기간: {period_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)

    # ── 중앙: (좌) 라인차트  ·  (우) 이상거래 리스트 ──
    left, right = st.columns([4, 1.8])

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
        st.markdown("</div>", unsafe_allow_html=True)   # ← div 닫기

st.markdown(
    """
    <style>
      /* ...여기 기존 스타일들... */

      /* 사이드바 page_link(버튼) 텍스트 보이게 */
      section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
      section[data-testid="stSidebar"] [data-testid^="stPageLink"] span {
        color:#EAF2FF !important;  /* 글자색 밝게 */
        opacity:1 !important;
      }
      /* 선택된 페이지(현재 페이지)도 가독성 유지 */
      section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-current="page"] span {
        color:#FFFFFF !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── 선택된 페이지 실행 (필수) ─────────────────────
current.run()
