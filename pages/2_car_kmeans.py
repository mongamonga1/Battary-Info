# -*- coding: utf-8 -*-
"""차명별 K-means 군집 분석"""
import warnings, re
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from math import pi
from itertools import cycle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─────────────────────────── 설정 ───────────────────────────
st.header("🚗 차명별 K-means 군집 분석")

# 1) 데이터 불러오기
DATA_PATH = Path("data/SoH_NCM_Dataset_selected_Fid_및_배터리등급열추가.xlsx")
uploaded  = st.sidebar.file_uploader("엑셀 업로드(선택)", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded, engine="openpyxl")
    st.success("업로드한 파일을 사용합니다.")
elif DATA_PATH.exists():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
else:
    st.error("기본 엑셀 파일(data/SoH_NCM_Dataset_selected.xlsx)을 찾을 수 없습니다. "
             "사이드바에서 파일을 업로드해 주세요.")
    st.stop()

# 2) 전처리
df.columns = df.columns.str.strip()
df.rename(columns={
    "사용연수(t)": "Age",
    "SoH_pred(%)": "SoH",
    "중고거래가격": "Price",
    "셀 간 균형": "CellBalance",
}, inplace=True)
df["CellBalance"] = df["CellBalance"].map(
    {"우수": "Good", "경고": "Warning", "심각": "Critical"}
)

num_cols = ["Age", "SoH", "Price"]
cat_col  = "CellBalance"

preproc = ColumnTransformer(
    [("num", StandardScaler(), num_cols),
     ("cat", OneHotEncoder(drop="first"), [cat_col])]
)

model_list = sorted(df["차명"].dropna().unique())
choice = st.sidebar.selectbox("차명 선택", model_list)

# 3) K-means 실행
sub = df[df["차명"] == choice].copy()
if len(sub) < 3:
    st.warning(f"{choice} 샘플이 {len(sub)}건으로 분석할 수 없습니다.")
    st.stop()

X = preproc.fit_transform(sub)
if hasattr(X, "toarray"):
    X = X.toarray()

ks = range(2, min(10, len(sub)))
sil_scores = [
    silhouette_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X))
    for k in ks
]
opt_k = ks[int(np.argmax(sil_scores))]

labels = KMeans(n_clusters=opt_k, random_state=42).fit_predict(X)
sub["cluster"] = labels
palette = cycle(sns.color_palette("tab10"))

# 4) 시각화 예시 ─ Boxplot
for col in num_cols:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="cluster", y=col, data=sub, palette="tab10", ax=ax)
    ax.set_title(f"{choice}: {col} by Cluster (k={opt_k})")
    st.pyplot(fig)

# 5) 2D PCA 시각화
pca2 = PCA(2, random_state=42).fit_transform(X)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(pca2[:, 0], pca2[:, 1], c=labels, cmap="tab10", s=60, edgecolors="k")
ax.set_title(f"{choice}: PCA 2-D (k={opt_k})")
st.pyplot(fig)
