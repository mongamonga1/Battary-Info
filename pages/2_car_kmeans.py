# -*- coding: utf-8 -*-
"""차명별 K-means 군집 분석 (k 자동선정, 모든 결과·프로파일 가로 스크롤)"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
from math import pi
from itertools import cycle
from io import BytesIO
import base64

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# SciPy / Yellowbrick → 내부 계산만(표시는 X)
try:
    from scipy.cluster.hierarchy import linkage
    _has_scipy = True
except Exception:
    _has_scipy = False

try:
    from yellowbrick.cluster import KElbowVisualizer
    _has_yb = True
except Exception:
    _has_yb = False

# ─────────────────────────── 설정 ───────────────────────────
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

st.header("🚗 차명별 K-means 군집 분석")

# ───────────────────────── 데이터 로드 ─────────────────────────
DATA_PATH = Path("data/SoH_NCM_Dataset_selected_Fid_및_배터리등급열추가.xlsx")
uploaded = st.sidebar.file_uploader("엑셀 업로드(선택)", type=["xlsx"])

def load_excel(path_or_buffer) -> pd.DataFrame:
    df = pd.read_excel(path_or_buffer, engine="openpyxl")
    df.columns = df.columns.map(lambda x: str(x).strip())
    return df

if uploaded:
    df_raw = load_excel(uploaded)
    st.success("업로드한 파일을 사용합니다.")
elif DATA_PATH.exists():
    df_raw = load_excel(DATA_PATH)
    st.info(f"기본 엑셀 사용: {DATA_PATH}")
else:
    st.error("기본 엑셀 파일을 찾을 수 없습니다. 사이드바에서 업로드해 주세요.")
    st.stop()

# ─────────────────────── 컬럼 표준화/중복 방지 ───────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def pick_first(cands):
        for c in cands:
            if c in out.columns:
                return c
        return None

    mapping = {}
    schema = [
        ("Model",       ["차명", "배터리종류", "차종", "모델"]),
        ("Age",         ["사용연수(t)", "사용연수", "연식"]),
        ("SoH",         ["SoH_pred(%)", "SoH(%)", "SOH"]),
        ("Price",       ["중고거래가격", "개당가격", "거래금액", "가격"]),
        ("CellBalance", ["셀 간 균형", "셀간균형"]),
    ]
    for std, cands in schema:
        src = pick_first(cands)
        if src:
            mapping[src] = std

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
            out["Price"].astype(str)
            .str.replace(r"[^\d.\-]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    if "SoH" in out.columns:
        out["SoH"] = pd.to_numeric(out["SoH"], errors="coerce")

    return out

df = normalize_columns(df_raw)

# 필수 컬럼/수치 컬럼
if "Model" not in df.columns:
    st.error("엑셀에 '차명/배터리종류/차종/모델' 중 하나가 없어 Model 컬럼을 만들 수 없습니다.")
    st.stop()

num_pool = [c for c in ["Age", "SoH", "Price"] if c in df.columns]
if len(num_pool) < 2:
    st.error(f"수치 컬럼이 부족합니다(필요≥2). 현재: {num_pool}")
    st.stop()

# ───────────────────────── 사이드바 ─────────────────────────
models        = sorted(df["Model"].dropna().astype(str).unique())
choice        = st.sidebar.selectbox("차명 선택", models)
show_tsne     = st.sidebar.checkbox("t-SNE 2D 추가", value=True)
show_pca3     = st.sidebar.checkbox("PCA 3D 추가", value=False)
perplexity    = st.sidebar.slider("t-SNE perplexity", 5, 50, 30, 1)
show_profiles = st.sidebar.checkbox("추가 프로파일(가로 스크롤)", value=True)

# ───────────────────────── 데이터 준비 ─────────────────────────
sub_all = df[df["Model"].astype(str) == str(choice)].copy()
sub_all = sub_all.dropna(subset=num_pool)
n = len(sub_all)
if n < 3:
    st.warning(f"'{choice}' 유효 표본이 {n}건이라 분석할 수 없습니다(≥3 필요).")
    st.stop()

ks = list(range(2, min(10, n)))

preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_pool),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         ["CellBalance"] if "CellBalance" in sub_all.columns else []),
    ],
    remainder="drop",
)
X = preproc.fit_transform(sub_all)
if hasattr(X, "toarray"):
    X = X.toarray()

# ───────────── k 선택: Silhouette + Elbow + Dendrogram → Median ─────────────
def choose_k_multi(X, ks):
    votes = {}
    # Silhouette
    try:
        sil_scores = [
            silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X))
            for k in ks if k < len(X)
        ]
        if sil_scores:
            votes["silhouette"] = ks[int(np.argmax(sil_scores))]
    except Exception:
        pass
    # Elbow
    try:
        if _has_yb:
            viz = KElbowVisualizer(KMeans(random_state=42), k=ks, metric="distortion", timings=False)
            viz.fit(X)
            if viz.elbow_value_ is not None:
                votes["elbow"] = int(viz.elbow_value_)
        else:
            inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X).inertia_ for k in ks]
            if len(inertias) >= 2:
                diffs = np.diff(inertias)
                idx = int(np.argmax(diffs))
                votes["elbow"] = ks[idx + 1] if idx + 1 < len(ks) else ks[-1]
    except Exception:
        pass
    # Dendrogram gap
    try:
        if _has_scipy:
            n = X.shape[0]
            idx = np.arange(n if n <= 200 else 200)
            Z = linkage(X[idx], method="ward")
            dists = Z[:, 2]; gaps = np.diff(dists)
            if len(gaps) >= 1:
                k_est = n - (int(np.argmax(gaps)) + 1)
                votes["dendrogram"] = max(2, min(k_est, ks[-1]))
    except Exception:
        pass

    vals = [v for v in [votes.get("silhouette"), votes.get("elbow"), votes.get("dendrogram")] if v is not None]
    k_final = int(np.median(vals)) if vals else 3
    return k_final, votes

k_final, votes = choose_k_multi(X, ks)
st.caption(f"선택된 k = {k_final} (Sil={votes.get('silhouette','—')}, "
           f"Elbow={votes.get('elbow','—')}, Dend={votes.get('dendrogram','—')} → median)")

# ───────────────────────── 학습 & 라벨 ─────────────────────────
labels = KMeans(n_clusters=k_final, random_state=42, n_init="auto").fit_predict(X)
sub_all = sub_all.copy()
sub_all["cluster"] = labels
clusters = sorted(sub_all["cluster"].unique())

# 공용: Matplotlib Figure → base64 PNG
def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# 공용: 가로 스크롤 컨테이너 스타일
st.markdown("""
<style>
.scroll-x { overflow-x: auto; padding: 8px 0 10px; }
.scroll-row { display: inline-flex; gap: 16px; }
.scroll-row img { border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
.caption-center { text-align:center; color: #6b7280; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── 결과 그래프(가로 스크롤) ─────────────────────────
main_figs = []

# PCA 2D
p2 = PCA(2, random_state=42).fit_transform(X)
f = plt.figure(figsize=(5.2, 4.0))
plt.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
plt.title(f"{choice}: PCA 2D (k={k_final})"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
main_figs.append(fig_to_base64(f))

# Radar(클러스터 평균, 0~1 정규화)
mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
norm_means = mean_matrix.copy()
for c in num_pool:
    mn, mx = df[c].min(), df[c].max()
    norm_means[c] = 0.5 if (pd.isna(mn) or pd.isna(mx) or mx == mn) else (norm_means[c] - mn) / (mx - mn)

angles = [i / len(num_pool) * 2 * pi for i in range(len(num_pool))] + [0]
f = plt.figure(figsize=(5.2, 4.0))
ax = plt.subplot(111, polar=True)
for i in clusters:
    vals = norm_means.loc[i].tolist(); vals.append(vals[0])
    ax.plot(angles, vals, label=f"Cluster {i}"); ax.fill(angles, vals, alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(num_pool)
plt.title(f"{choice}: Radar (k={k_final})")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
main_figs.append(fig_to_base64(f))

# t-SNE 2D (옵션: 오른쪽에 추가)
if show_tsne:
    perp = min(perplexity, n - 1)
    ts2 = TSNE(n_components=2, perplexity=perp, max_iter=500, random_state=42, init="pca").fit_transform(X)
    f = plt.figure(figsize=(5.2, 4.0))
    plt.scatter(ts2[:, 0], ts2[:, 1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
    plt.title(f"{choice}: t-SNE 2D (k={k_final})"); plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2"); plt.tight_layout()
    main_figs.append(fig_to_base64(f))

# PCA 3D (옵션: 오른쪽에 추가)
if show_pca3:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    p3 = PCA(3, random_state=42).fit_transform(X)
    f = plt.figure(figsize=(5.6, 4.2))
    ax3 = f.add_subplot(111, projection="3d")
    ax3.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=labels, cmap="tab10", s=50, edgecolors="k", alpha=0.85)
    ax3.set_title(f"{choice}: PCA 3D (k={k_final})")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    main_figs.append(fig_to_base64(f))

# 출력: 메인 결과 가로 스크롤
html_main = "".join([f"<img src='data:image/png;base64,{b}' height='320'/>" for b in main_figs])
st.markdown(f"<div class='scroll-x'><div class='scroll-row'>{html_main}</div></div>", unsafe_allow_html=True)
st.markdown("<div class='caption-center'>좌우 스크롤로 모든 결과 그래프(PCA 2D, Radar, 옵션: t-SNE/PCA 3D)를 확인하세요.</div>", unsafe_allow_html=True)

# ───────────────────────── 추가 프로파일(가로 스크롤) ─────────────────────────
if show_profiles:
    figs = []

    # 1) Boxplots (바이올린 차트 제거)
    for col in num_pool:
        f = plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=col, data=sub_all, palette="tab10")
        plt.title(f"{choice}: {col} by Cluster (k={k_final})")
        figs.append(fig_to_base64(f))

    # 2) 범주 Count + Stacked(%) Bar (있을 때)
    if "CellBalance" in sub_all.columns:
        f = plt.figure(figsize=(6, 4))
        sns.countplot(x="cluster", hue="CellBalance", data=sub_all, palette="Set2")
        plt.title(f"{choice}: Count of CellBalance by Cluster")
        figs.append(fig_to_base64(f))

        ctab_pct = pd.crosstab(sub_all["cluster"], sub_all["CellBalance"], normalize="index") * 100
        ctab_pct = ctab_pct.reindex(clusters, fill_value=0)
        f = plt.figure(figsize=(6, 4))
        ax2 = plt.gca()
        ctab_pct.plot(kind="bar", stacked=True, colormap="Paired", ax=ax2)
        plt.title(f"{choice}: CellBalance Distribution (%) by Cluster")
        plt.tight_layout()
        figs.append(fig_to_base64(f))

    # 3) Heatmap of means
    mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
    f = plt.figure(figsize=(6, 4))
    sns.heatmap(mean_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{choice}: Numeric Feature Means per Cluster")
    figs.append(fig_to_base64(f))

    # 출력: 추가 프로파일 가로 스크롤
    html_prof = "".join([f"<img src='data:image/png;base64,{b}' height='300'/>" for b in figs])
    st.markdown(f"<div class='scroll-x'><div class='scroll-row'>{html_prof}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='caption-center'>가로 스크롤(드래그바)을 좌우로 움직여 모든 추가 프로파일을 확인하세요.</div>", unsafe_allow_html=True)
