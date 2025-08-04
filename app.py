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
    "reco": ROOT / "pages/4_recommend_system.py",
    "fraud": ROOT / "pages/5_forest_lstm.py",
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
            .resample("ME")  # ← 'M' 대신 'ME' (FutureWarning 대응)
            .size()
            .rename("count")
            .reset_index()
        )
        fig_line = px.line(monthly_cnt, x="계약일", y="count", markers=True)
        fig_line.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="box"><div class="box-title">🚨 이상거래 의심 내역</div>', unsafe_allow_html=True)
        if "개당가격" in df.columns:
            df2 = df.sort_values("계약일").copy()
            # ← pct_change 기본값 deprec. → fill_method=None 명시
            df2["변동"] = pd.to_numeric(df2["개당가격"], errors="coerce").pct_change(fill_method=None).fillna(0)
            label_col = next((c for c in ["배터리종류", "모델", "차종", "판매업체"] if c in df2.columns), df2.columns[0])
            top_issue = (
                df2.tail(40)
                .nlargest(6, "변동")
                .assign(
                    change=lambda d: (d["변동"] * 100).round(2),
                    price=lambda d: d["개당가격"].map(lambda x: f"₩ {x:,.0f}"),
                )
            )
            low_issue = (
                df2.tail(40)
                .nsmallest(6, "변동")
                .assign(
                    change=lambda d: (d["변동"] * 100).round(2),
                    price=lambda d: d["개당가격"].map(lambda x: f"₩ {x:,.0f}"),
                )
            )
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
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("가격 컬럼이 없어 최근 거래 기준의 단순 목록만 표시합니다.")
            for s in df.head(9).index:
                st.markdown(f"- 항목 {s}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 하단: (좌) 고객 지원 · (우) 분석 결과 ──
    c_left, c_right = st.columns([2.4, 2])

    with c_left:
        st.markdown('<div class="box"><div class="box-title">🧑‍💼 고객 지원</div>', unsafe_allow_html=True)
        demo_support = pd.DataFrame(
            {
                "Date": [
                    today.strftime("%Y/%m/%d %H:%M:%S"),
                    (today - pd.Timedelta("1D")).strftime("%Y/%m/%d %H:%M:%S"),
                    (today - pd.Timedelta("2D")).strftime("%Y/%m/%d %H:%M:%S"),
                ],
                "제목": ["이상거래 의심 제보", "이상거래 소명", "데이터 정합성 문의"],
                "사용자": ["이**(d****)", "김**(f******)", "박**(k*****)"],
            }
        )
        st.dataframe(demo_support, use_container_width=True, height=240)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── (우) 차명별 군집 결과 – 엑셀 기반 간단 프리뷰 ──
    #   홈 화면에서도 엑셀이 있으면 차명별 KMeans 결과 요약을 보여줍니다.
    with c_right:
        st.markdown('<div class="box"><div class="box-title">📌 차명별 군집 결과</div>', unsafe_allow_html=True)

        # 1) 엑셀 로드
        KMEANS_PATH = ROOT / "data/SoH_NCM_Dataset_selected_Fid_및_배터리등급열추가.xlsx"

        @st.cache_data(show_spinner=False)
        def load_kmeans_data(path: Path) -> pd.DataFrame | None:
            if not path.exists():
                return None
            dfk = pd.read_excel(path, sheet_name=0, engine="openpyxl")
            dfk.columns = dfk.columns.map(lambda x: str(x).strip())
            return dfk

        df_kmeans = load_kmeans_data(KMEANS_PATH)

        # 2) 전처리/모델링 유틸
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        try:
            from scipy.cluster.hierarchy import linkage
            _has_scipy = True
        except Exception:
            _has_scipy = False

        def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()

            def pick_first(candidates):
                for c in candidates:
                    if c in out.columns:
                        return c
                return None

            mapping = {}
            schema = [
                ("Model", ["차명", "배터리종류", "차종", "모델"]),
                ("Age", ["사용연수(t)", "사용연수", "연식"]),
                ("SoH", ["SoH_pred(%)", "SoH(%)", "SOH"]),
                ("Price", ["중고거래가격", "개당가격", "거래금액", "가격"]),
                ("CellBalance", ["셀 간 균형", "셀간균형"]),
            ]
            for std_name, candidates in schema:
                c = pick_first(candidates)
                if c:
                    mapping[c] = std_name

            out = out.rename(columns=mapping)

            if out.columns.duplicated().any():
                out = out.loc[:, ~out.columns.duplicated()]

            if "CellBalance" in out.columns:
                out["CellBalance"] = (
                    out["CellBalance"]
                    .map({"우수": "Good", "정상": "Normal", "경고": "Warning", "심각": "Critical"})
                    .fillna(out["CellBalance"])
                )

            if "Price" in out.columns:
                out["Price"] = (
                    out["Price"]
                    .astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                    .pipe(pd.to_numeric, errors="coerce")
                )
            if "Age" in out.columns:
                out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
            if "SoH" in out.columns:
                out["SoH"] = pd.to_numeric(out["SoH"], errors="coerce")

            return out

        def _choose_k_multi(X, ks, max_dendro_samples: int = 200):
            results = {}
            # Silhouette
            try:
                sil_scores = []
                for k in ks:
                    if k >= len(X):
                        break
                    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
                    sil_scores.append(silhouette_score(X, labels))
                if sil_scores:
                    results["silhouette"] = ks[int(np.argmax(sil_scores))]
            except Exception:
                pass

            # Elbow(Inertia)
            try:
                inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X).inertia_ for k in ks]
                if len(inertias) >= 2:
                    diffs = np.diff(inertias)
                    results["elbow"] = ks[int(np.argmax(diffs)) + 1]
            except Exception:
                pass

            # Dendrogram gap (Ward)
            try:
                if _has_scipy:
                    n = X.shape[0]
                    idx = np.arange(n)
                    if n > max_dendro_samples:
                        idx = np.random.choice(n, max_dendro_samples, replace=False)
                    Z = linkage(X[idx], method="ward")
                    dists = Z[:, 2]
                    gaps = np.diff(dists)
                    if len(gaps) >= 1:
                        k_dend = max(2, min(n - (int(np.argmax(gaps)) + 1), ks[-1]))
                        results["dendrogram"] = k_dend
            except Exception:
                pass

            votes = [results.get("silhouette"), results.get("elbow"), results.get("dendrogram")]
            votes = [v for v in votes if v is not None]
            if not votes:
                return {"k_final": 3, "detail": results}

            k_final = int(np.median(votes))
            results["k_final"] = k_final
            results["detail"] = results.copy()
            return results

        def make_model_charts(
            df: pd.DataFrame,
            model_name: str,
            k: int | str = "auto",  # "auto" → 위의 멀티 방식 사용
            reducer: str = "pca",
            aggregate_radar: bool = True,  # 메인에는 평균 1개 레이더가 깔끔
        ):
            df = _normalize_columns(df)

            # 필수 컬럼 체크
            if "Model" not in df.columns:
                raise ValueError("필수 컬럼 'Model'이 없습니다.")

            # 사용 가능한 수치 컬럼(최소 2개 필요)
            numeric_pool = [c for c in ["Age", "SoH", "Price"] if c in df.columns]
            if len(numeric_pool) < 2:
                raise ValueError(f"수치 컬럼이 부족합니다(필요≥2): {numeric_pool}")

            # 모델 필터 + 수치 결측 제거
            sub = df[df["Model"].astype(str).str.contains(model_name, case=False, na=False)].copy()
            sub = sub.dropna(subset=numeric_pool)
            if sub.empty or len(sub) < 3:
                raise ValueError(f"'{model_name}' 유효 데이터가 {len(sub)}건입니다(≥3 필요).")

            # 혹시 모를 중복 열 제거
            if sub.columns.duplicated().any():
                sub = sub.loc[:, ~sub.columns.duplicated()]

            # 전처리 파이프라인
            pre = ColumnTransformer(
                [
                    ("num", StandardScaler(), numeric_pool),
                    (
                        "cat",
                        OneHotEncoder(drop="first", handle_unknown="ignore"),
                        ["CellBalance"] if "CellBalance" in sub.columns else [],
                    ),
                ],
                remainder="drop",
            )

            X = pre.fit_transform(sub)
            if hasattr(X, "toarray"):
                X = X.toarray()

            # k 결정
            if isinstance(k, str) and k == "auto":
                ks = list(range(2, min(10, len(sub))))  # 2 ~ 9 (또는 최대 n-1)
                choose = _choose_k_multi(X, ks)
                k_final = int(choose["k_final"])
                k_detail = choose["detail"] if "detail" in choose else {}
            else:
                k_final = int(k)
                k_detail = {}

            labels = KMeans(n_clusters=k_final, random_state=42, n_init="auto").fit_predict(X)
            sub["cluster"] = labels
            clusters = sorted(sub["cluster"].unique())

            # ── 레이더(0~100 정규화, Age는 낮을수록 좋다고 가정해 뒤집기) ──
            scaler = MinMaxScaler(feature_range=(0, 100))
            norm_vals = pd.DataFrame(scaler.fit_transform(sub[numeric_pool]), columns=numeric_pool, index=sub.index)
            if "Age" in norm_vals.columns:
                norm_vals["Age"] = 100 - norm_vals["Age"]

            radar_fig = go.Figure()
            if aggregate_radar:
                avg = norm_vals.mean().reindex(numeric_pool).tolist()
                radar_fig.add_trace(
                    go.Scatterpolar(r=avg + [avg[0]], theta=numeric_pool + [numeric_pool[0]], fill="toself", name=model_name)
                )
            else:
                for c in clusters:
                    v = norm_vals.loc[sub["cluster"] == c, numeric_pool].mean().tolist()
                    radar_fig.add_trace(
                        go.Scatterpolar(r=v + [v[0]], theta=numeric_pool + [numeric_pool[0]], fill="toself", name=f"Cluster {c}")
                    )
            radar_fig.update_layout(
                title=f"{model_name} : Radar",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                margin=dict(l=10, r=10, t=30, b=10),
                height=260,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )

            # ── 산점도(PCA 2D) ──
            if reducer == "pca":
                from sklearn.decomposition import PCA as _PCA

                pts = _PCA(n_components=2, random_state=42).fit_transform(X)
                xlab, ylab = "PC1", "PC2"
            else:
                pts = np.c_[np.arange(len(sub)), np.zeros(len(sub))]
                xlab, ylab = "index", ""
            import plotly.express as _px

            scatter_fig = _px.scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                color=sub["cluster"].astype(str),
                labels={"x": xlab, "y": ylab, "color": "Cluster"},
                title=f"{model_name} : Cluster Scatter ({'PCA 2D' if reducer=='pca' else 'index'})",
                height=280,
            )
            scatter_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))

            return radar_fig, scatter_fig, k_final, k_detail

        # 3) 렌더링
        if df_kmeans is None:
            st.info("KMeans용 엑셀을 찾을 수 없습니다. `data/SoH_NCM_Dataset_selected_Fid_및_배터리등급열추가.xlsx` 를 넣어주세요.")
        else:
            model_col = "차명" if "차명" in df_kmeans.columns else ("Model" if "Model" in df_kmeans.columns else None)
            if model_col is None:
                st.warning("엑셀에 '차명' 또는 'Model' 컬럼이 없습니다.")
            else:
                models = sorted(df_kmeans[model_col].dropna().astype(str).unique())
                pick = st.selectbox("차종 선택", models, index=0 if models else None, label_visibility="collapsed")
                if pick:
                    try:
                        radar_fig, scatter_fig, k_final, k_detail = make_model_charts(
                            df_kmeans, model_name=str(pick), k="auto", reducer="pca", aggregate_radar=True
                        )
                        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
                        st.plotly_chart(scatter_fig, use_container_width=True, config={"displayModeBar": False})

                        det = k_detail
                        sil = det.get("silhouette", "—")
                        elb = det.get("elbow", "—")
                        den = det.get("dendrogram", "—")
                        st.caption(f"선택된 k = {k_final} (Sil={sil}, Elbow={elb}, Dend={den} → median).")

                    except Exception as e:
                        st.warning(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    # ── 데이터 미리보기 ──
    st.markdown('<div class="blank"></div>', unsafe_allow_html=True)
    with st.expander("데이터 미리보기 (앞 50행)"):
        st.dataframe(df.head(50), use_container_width=True)

    st.caption("© 2025 Battery-Info ― 사이드바 커스텀 메뉴에서 상세 분석 페이지로 이동하세요.")


# ───────────────────── 페이지 등록 ─────────────────────
home = st.Page(render_home, title="🏠 홈", default=True, url_path="")
pg_kmeans = st.Page(str(PAGES["kmeans"]), title="🚗 차명별 군집분석", url_path="kmeans")
pg_reco = st.Page(str(PAGES["reco"]), title="✨ 기업 추천", url_path="reco")
pg_fraud = st.Page(str(PAGES["fraud"]), title="🌳 이상거래 의심", url_path="fraud")
pg_ts = st.Page(str(PAGES["timeseries"]), title="📈 시세 분석", url_path="timeseries")

# 숨김 내비 생성
current = st.navigation([home, pg_kmeans, pg_reco, pg_fraud, pg_ts], position="hidden")

# ───────────────────── 사이드바(공통) ─────────────────────
with st.sidebar:
    # 상단 브랜드 영역(그대로 유지)
    st.markdown(
        '<div style="position:sticky;top:0;z-index:10;background:#0f1b2d;padding:12px 12px 6px;'
        'margin:0 -8px 8px -8px;border-bottom:1px solid rgba(255,255,255,.06);">'
        '<div style="font-weight:900;font-size:24px;letter-spacing:.8px;color:#fff;line-height:1.2;">BATTERY-INFO</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ⬇️ 메인 화면으로 가는 링크(요거 추가)
    st.page_link(home, label="메인 화면", icon="🏠")

    # 상세 페이지들은 접히는 그룹으로
    with st.expander("분석 결과 확인", expanded=True):
        st.page_link(pg_kmeans, label="군집 분석", icon="🚗")
        st.page_link(pg_reco,   label="기업 추천", icon="✨")
        st.page_link(pg_fraud,  label="이상거래 의심", icon="🌳")
        st.page_link(pg_ts,     label="시세 분석", icon="📈")

# ───────────────────── 선택된 페이지 실행 (필수) ─────────────────────
current.run()
