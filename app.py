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
# 사이드바 상단 브랜드(크게) + 고정
st.markdown("""
<style>
  [data-testid="stSidebar"] .brand-wrap{
    position: sticky; top: 0; z-index: 10;
    background:#0f1b2d;               /* 사이드바 배경색과 동일 */
    padding:12px 12px 6px; margin:0 -8px 8px -8px;
    border-bottom:1px solid rgba(255,255,255,.06);
  }
  [data-testid="stSidebar"] .brand-title{
    font-weight: 900;
    font-size: 24px;                   /* ← 크게 보이게 */
    letter-spacing: .8px;
    color:#ffffff;
    line-height: 1.2;
  }
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<style>
  [data-testid="stSidebar"] .brand-wrap{
    position: sticky; top: 0; z-index: 10;
    background:#0f1b2d;               /* 사이드바 배경에 맞춤 */
    padding:10px 12px 8px; margin:0 -8px 8px -8px;
    border-bottom:1px solid rgba(255,255,255,.06);
  }
  [data-testid="stSidebar"] .brand{
    font-weight:900; font-size:18px; letter-spacing:.6px; color:#ffffff;
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
    # 상단 고정 브랜드
    st.markdown(
        '<div class="brand-wrap"><div class="brand-title">BATTERY-INFO</div></div>',
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("### 📂 분석 결과 확인", help="상단 기본 Pages 네비 대신 커스텀 메뉴를 사용합니다.")
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
# ──────────────────────────────────────────────────────────────
# 필요한 패키지
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# 한글 컬럼 → 영문 표준화(있을 때만)
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        '사용연수(t)': 'Age',
        'SoH_pred(%)': 'SoH',
        '중고거래가격': 'Price',
        '셀 간 균형': 'CellBalance',
        '차명': 'Model'
    }
    out = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}).copy()
    if 'CellBalance' in out.columns:
        out['CellBalance'] = out['CellBalance'].map({'우수':'Good','경고':'Warning','심각':'Critical'}).fillna(out['CellBalance'])
    return out

def _auto_k(X, ks):
    # 실루엣 최고값 k (계산 안 되면 3)
    try:
        scores = []
        for k in ks:
            if k >= len(X): break
            labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X)
            scores.append(silhouette_score(X, labels))
        return ks[int(np.argmax(scores))] if scores else 3
    except Exception:
        return 3

def make_model_charts(
    df: pd.DataFrame,
    model_name: str,
    k: int | str = "auto",
    reducer: str = "pca",
    aggregate_radar: bool = False,
):
    """
    df      : 원본 데이터프레임(한글/영문 컬럼 모두 허용)
    model_name : 차명 (예: '코나 일렉트릭 (KONA ELECTRIC)' 또는 'Kona Electric')
    k       : 군집 수 (정수 또는 'auto')
    reducer : 'pca' (권장) / 'none'
    aggregate_radar : True면 클러스터별 대신 '모델 평균 1개' 레이더를 그림
    반환     : (radar_fig: go.Figure, scatter_fig: go.Figure)
    """
    df = _normalize_columns(df)

    required = {'Model','Age','SoH','Price'}
    if not required.issubset(df.columns):
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {required - set(df.columns)}")

    sub = df[df['Model'].astype(str).str.contains(model_name, case=False, na=False)].copy()
    n = len(sub)
    if n < 3:
        raise ValueError(f"'{model_name}' 데이터가 {n}건으로 너무 적습니다(≥3 필요).")

    # 전처리
    num_cols = ['Age','SoH','Price']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['CellBalance'] if 'CellBalance' in sub.columns else [])
    ], remainder='drop')

    X = pre.fit_transform(sub)
    if hasattr(X, "toarray"):
        X = X.toarray()

    # k 결정
    if isinstance(k, str) and k == "auto":
        ks = list(range(2, min(9, n)))  # 2~8
        k_final = _auto_k(X, ks)
    else:
        k_final = int(k)

    # KMeans 라벨
    labels = KMeans(n_clusters=k_final, random_state=42, n_init='auto').fit_predict(X)
    sub['cluster'] = labels
    clusters = sorted(sub['cluster'].unique())

    # ── (1) 레이더 차트 ─────────────────────────────────────────
    # 수치 컬럼을 0~100으로 정규화 (모델 내 기준)
    scaler = MinMaxScaler(feature_range=(0, 100))
    norm_vals = pd.DataFrame(scaler.fit_transform(sub[num_cols]), columns=num_cols, index=sub.index)
    # 'Age'는 낮을수록 좋은 지표라고 가정 → 뒤집기(옵션)
    norm_vals['Age'] = 100 - norm_vals['Age']

    if aggregate_radar:
        # 모델 평균 1개 폴리곤
        avg = norm_vals.mean().reindex(num_cols).tolist()
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=avg + [avg[0]],
            theta=num_cols + [num_cols[0]],
            fill='toself',
            name=model_name
        ))
    else:
        # 클러스터별 폴리곤
        radar_fig = go.Figure()
        for c in clusters:
            v = norm_vals.loc[sub['cluster']==c, num_cols].mean().tolist()
            radar_fig.add_trace(go.Scatterpolar(
                r=v + [v[0]],
                theta=num_cols + [num_cols[0]],
                fill='toself',
                name=f'Cluster {c}'
            ))

    radar_fig.update_layout(
        title=f"{model_name} : Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )

    # ── (2) 산점도 (PCA 2D) ────────────────────────────────────
    if reducer == "pca":
        pts = PCA(n_components=2, random_state=42).fit_transform(X)
        scatter_fig = px.scatter(
            x=pts[:,0], y=pts[:,1],
            color=sub['cluster'].astype(str),
            labels={'x':'PC1','y':'PC2','color':'Cluster'},
            title=f"{model_name} : Cluster Scatter (PCA 2D)",
            height=280
        )
    else:
        scatter_fig = px.scatter(
            x=np.arange(n), y=np.zeros(n),
            color=sub['cluster'].astype(str),
            labels={'x':'index','y':'','color':'Cluster'},
            title=f"{model_name} : Clusters",
            height=280
        )

    scatter_fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return radar_fig, scatter_fig
# ──────────────────────────────────────────────────────────────

with c_right:
    st.markdown('<div class="box"><div class="box-title">📌 차명별 군집 결과</div>', unsafe_allow_html=True)

    # '차명' 또는 'Model' 컬럼 자동 감지
    model_col = '차명' if '차명' in df.columns else ('Model' if 'Model' in df.columns else None)

    if model_col:
        models = sorted(df[model_col].dropna().astype(str).unique())
        # 상단 드롭다운(숨김 라벨)
        pick = st.selectbox("차종 선택", models, index=0 if models else None, label_visibility="collapsed")

        if pick:
            try:
                radar_fig, scatter_fig = make_model_charts(
                    df,
                    model_name=str(pick),  # 선택한 차종
                    k="auto",              # 군집 수 자동 결정
                    reducer="pca",
                    aggregate_radar=True   # 평균 1개 레이더; 클러스터별이면 False
                )
                st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                st.warning(str(e))
    else:
        st.info("데이터에 '차명' 또는 'Model' 컬럼이 없어 차종을 선택할 수 없습니다.")

    st.markdown('</div>', unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────


# ───────────────────── 데이터 미리보기 ─────────────────────
st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
with st.expander("데이터 미리보기 (앞 50행)"):
    st.dataframe(df.head(50), use_container_width=True)

st.caption("© 2025 Battery-Info ― 사이드바 커스텀 메뉴에서 상세 분석 페이지로 이동하세요.")
