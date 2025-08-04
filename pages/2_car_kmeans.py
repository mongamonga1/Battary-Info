# -*- coding: utf-8 -*-
"""차명별 K-means 군집 분석 (k 자동선정, 결과 그래프 가로 배치)"""
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

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# SciPy(덴드로그램) / Yellowbrick(엘보우) 설치 여부만 확인 → 계산에만 사용(표시는 안 함)
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

    # 동일 이름 열 중복 제거
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    # 범주 매핑
    if "CellBalance" in out.columns:
        out["CellBalance"] = (
            out["CellBalance"]
            .map({"우수": "Good", "정상": "Normal", "경고": "Warning", "심각": "Critical"})
            .fillna(out["CellBalance"])
        )

    # 숫자 정리
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

# 필수 컬럼 체크
if "Model" not in df.columns:
    st.error("엑셀에 '차명/배터리종류/차종/모델' 중 하나가 없어 Model 컬럼을 만들 수 없습니다.")
    st.stop()

# 사용 가능한 수치 컬럼(최소 2개 권장)
num_pool = [c for c in ["Age", "SoH", "Price"] if c in df.columns]
if len(num_pool) < 2:
    st.error(f"수치 컬럼이 부족합니다(필요≥2). 현재: {num_pool}")
    st.stop()

# ───────────────────────── 사이드바 설정 ─────────────────────────
models = sorted(df["Model"].dropna().astype(str).unique())
choice = st.sidebar.selectbox("차명 선택", models)

show_profiles = st.sidebar.checkbox("추가 프로파일(박스/바이올린/히트맵/레이더)", value=True)
show_pca3     = st.sidebar.checkbox("PCA 3D 표시 (별도)", value=False)
show_tsne     = st.sidebar.checkbox("t-SNE 2D 표시", value=True)   # 기본 켬
perplexity    = st.sidebar.slider("t-SNE perplexity", min_value=5, max_value=50, value=30, step=1)
cols_per_row  = st.sidebar.slider("결과 그래프 가로 배치 수", 2, 3, 3)

# 후보 k 범위
sub_all = df[df["Model"].astype(str) == str(choice)].copy()
sub_all = sub_all.dropna(subset=num_pool)
n = len(sub_all)
if n < 3:
    st.warning(f"'{choice}' 유효 표본이 {n}건이라 분석할 수 없습니다(≥3 필요).")
    st.stop()
ks = list(range(2, min(10, n)))  # 2~9 또는 n-1

# 전처리 파이프라인
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
    # 1) Silhouette
    try:
        sil_scores = [
            silhouette_score(X, KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X))
            for k in ks if k < len(X)
        ]
        if sil_scores:
            k_sil = ks[int(np.argmax(sil_scores))]
            votes["silhouette"] = k_sil
    except Exception:
        sil_scores = None

    # 2) Elbow(Inertia)
    try:
        if _has_yb:
            viz = KElbowVisualizer(KMeans(random_state=42), k=ks, metric="distortion", timings=False)
            viz.fit(X)
            k_elbow = int(viz.elbow_value_) if viz.elbow_value_ is not None else None
        else:
            inertias = [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X).inertia_ for k in ks]
            diffs = np.diff(inertias)
            idx = int(np.argmax(diffs)) if len(diffs) else 0
            k_elbow = ks[idx + 1] if idx + 1 < len(ks) else ks[-1]
        if k_elbow:
            votes["elbow"] = k_elbow
    except Exception:
        k_elbow = None

    # 3) Dendrogram gap
    try:
        if _has_scipy:
            n = X.shape[0]
            idx = np.arange(n)
            if n > 200:
                idx = np.random.choice(n, 200, replace=False)
            from scipy.cluster.hierarchy import linkage  # 재확인
            Z = linkage(X[idx], method="ward")
            dists = Z[:, 2]
            gaps = np.diff(dists)
            if len(gaps) >= 1:
                k_est = n - (int(np.argmax(gaps)) + 1)
                k_dend = max(2, min(k_est, ks[-1]))
                votes["dendrogram"] = k_dend
    except Exception:
        pass

    # 최종 k = 존재하는 값들의 중앙값
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
palette = cycle(sns.color_palette("tab10"))

# ───────────────────────── 결과 그래프(가로 배치) ─────────────────────────
result_figs = []

# 1) PCA 2D
p2 = PCA(2, random_state=42).fit_transform(X)
fig_pca = plt.figure(figsize=(4.6, 3.8))
plt.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
plt.title(f"{choice}: PCA 2D (k={k_final})")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
result_figs.append(("PCA 2D", fig_pca))

# 2) t-SNE 2D (옵션)
if show_tsne:
    perp = min(perplexity, n - 1)
    ts2 = TSNE(n_components=2, perplexity=perp, max_iter=500, random_state=42, init="pca").fit_transform(X)
    fig_ts2 = plt.figure(figsize=(4.6, 3.8))
    plt.scatter(ts2[:, 0], ts2[:, 1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
    plt.title(f"{choice}: t-SNE 2D (k={k_final})")
    plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2"); plt.tight_layout()
    result_figs.append(("t-SNE 2D", fig_ts2))

# 3) Radar(클러스터 평균, 0~1 정규화)
mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
norm_means = mean_matrix.copy()
for c in num_pool:
    mn, mx = df[c].min(), df[c].max()
    if pd.notna(mn) and pd.notna(mx) and mx != mn:
        norm_means[c] = (norm_means[c] - mn) / (mx - mn)
    else:
        norm_means[c] = 0.5  # 안전값

angles = [i / len(num_pool) * 2 * pi for i in range(len(num_pool))] + [0]
fig_radar = plt.figure(figsize=(4.6, 3.8))
ax = plt.subplot(111, polar=True)
for i in clusters:
    vals = norm_means.loc[i].tolist()
    vals.append(vals[0])
    ax.plot(angles, vals, label=f"Cluster {i}")
    ax.fill(angles, vals, alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(num_pool)
plt.title(f"{choice}: Radar (k={k_final})")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
result_figs.append(("Radar", fig_radar))

# 가로(타일) 배치 출력
cols = st.columns(cols_per_row)
for i, (_, fig) in enumerate(result_figs):
    with cols[i % cols_per_row]:
        st.pyplot(fig, use_container_width=True)
    # 줄바꿈
    if (i + 1) % cols_per_row == 0 and (i + 1) < len(result_figs):
        cols = st.columns(cols_per_row)

# (선택) PCA 3D는 별도 아래에 단독 표시
if show_pca3:
    p3 = PCA(3, random_state=42).fit_transform(X)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=labels, cmap="tab10", s=50, edgecolors="k", alpha=0.85)
    ax3.set_title(f"{choice}: PCA 3D (k={k_final})")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    st.pyplot(fig3)

# ───────────────────────── 추가 프로파일(옵션, 세로) ─────────────────────────
if show_profiles:
    # Box & Violin
    for col in num_pool:
        fig = plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=col, data=sub_all, palette="tab10")
        plt.title(f"{choice}: {col} by Cluster (k={k_final})")
        st.pyplot(fig)

        fig = plt.figure(figsize=(6, 4))
        sns.violinplot(x="cluster", y=col, data=sub_all, palette="tab10", inner="quartile")
        plt.title(f"{choice}: {col} Violin by Cluster")
        st.pyplot(fig)

    # 범주 Count + Stacked Bar (CellBalance가 있을 때)
    if "CellBalance" in sub_all.columns:
        fig = plt.figure(figsize=(6, 4))
        sns.countplot(x="cluster", hue="CellBalance", data=sub_all, palette="Set2")
        plt.title(f"{choice}: Count of CellBalance by Cluster")
        st.pyplot(fig)

        ctab_pct = pd.crosstab(sub_all["cluster"], sub_all["CellBalance"], normalize="index") * 100
        ctab_pct = ctab_pct.reindex(clusters, fill_value=0)
        fig = plt.figure(figsize=(6, 4))
        ctab_pct.plot(kind="bar", stacked=True, colormap="Paired", ax=plt.gca())
        plt.title(f"{choice}: CellBalance Distribution (%) by Cluster")
        plt.tight_layout()
        st.pyplot(fig)

    # Heatmap of means
    mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(mean_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{choice}: Numeric Feature Means per Cluster")
    st.pyplot(fig)

    # 통계 표
    stats = sub_all.groupby("cluster")[num_pool].agg(["count", "mean", "std", "min", "max", "median"])
    st.subheader("클러스터 통계 요약")
    st.dataframe(stats)

    # 텍스트 요약
    st.subheader("텍스트 요약")
    cluster_pct = sub_all["cluster"].value_counts(normalize=True).reindex(clusters, fill_value=0) * 100
    means = sub_all.groupby("cluster")[num_pool].mean().reindex(clusters)
    if "CellBalance" in sub_all.columns:
        ctab_pct = pd.crosstab(sub_all["cluster"], sub_all["CellBalance"], normalize="index") * 100
        ctab_pct = ctab_pct.reindex(clusters, fill_value=0)
    for i in clusters:
        if "CellBalance" in sub_all.columns and not ctab_pct.loc[i].empty:
            dom = ctab_pct.loc[i].idxmax()
            dom_val = ctab_pct.loc[i].max()
            dom_txt = f", dominant CellBalance '{dom}' ({dom_val:.1f}%)"
        else:
            dom_txt = ""
        st.write(
            f"- Cluster {i}: {cluster_pct[i]:.1f}% samples, "
            + ", ".join([f"avg {c} {means.loc[i, c]:.2f}" for c in num_pool])
            + dom_txt
        )
