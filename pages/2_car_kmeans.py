# -*- coding: utf-8 -*-
"""차명별 K-means 군집 분석 (k 자동선정, 가로 스크롤, 저가 GPT 요약/Word Export - Chat Completions)"""
import warnings
warnings.filterwarnings("ignore")

import os, base64
from io import BytesIO
from pathlib import Path
from math import pi
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- OpenAI secrets 헬퍼: [openai] 섹션/단일 키/환경변수 모두 지원 ---
def get_openai_conf():
    """
    반환: (api_key:str|None, model_name:str|None)
    - Streamlit Secrets의 [openai].api_key / [openai].model 우선
    - OPENAI_API_KEY(단일 키)도 허용
    - 환경변수 OPENAI_API_KEY 도 마지막 보루
    """
    api_key = None
    model_name = None

    # 1) [openai] 섹션
    if hasattr(st, "secrets") and "openai" in st.secrets:
        sect = st.secrets["openai"]
        api_key = sect.get("api_key") or api_key
        model_name = sect.get("model") or model_name

    # 2) 단일 키도 허용
    if hasattr(st, "secrets"):
        api_key = api_key or st.secrets.get("OPENAI_API_KEY")

    # 3) 환경변수
    import os
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    return api_key, model_name

# ── 선택 라이브러리(있으면 활용: 덴드로그램/엘보우)
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

# ── Word Export
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml.ns import qn

# ── OpenAI (Chat Completions)
try:
    from openai import OpenAI
    _has_openai = True
except Exception:
    _has_openai = False

# ───────────────────── 기본 설정 ─────────────────────
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False
st.header("🚗 차명별 K-means 군집 분석")

# ───────────────────── 데이터 로드 ─────────────────────
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
else:
    st.error("기본 엑셀 파일을 찾을 수 없습니다. 사이드바에서 업로드해 주세요.")
    st.stop()

# ───────────────────── 컬럼 표준화 ─────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def pick_first(cands):
        for c in cands:
            if c in out.columns: return c
        return None
    mapping = {}
    schema = [
        ("Model",       ["차명", "배터리종류", "차종", "모델"]),
        ("Age",         ["사용연수(t)", "사용연수", "연식"]),
        ("SoH",         ["SoH_pred(%)", "SoH(%)", "SOH"]),
        ("Price",       ["중고거래가격", "개당가격", "거래금액", "가격"]),
        ("CellBalance", ["셀 간 균형", "셀간균형"]),
    ]
    for std,cands in schema:
        src = pick_first(cands)
        if src: mapping[src] = std
    out = out.rename(columns=mapping)

    # 중복 열 제거
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    # 범주 매핑
    if "CellBalance" in out.columns:
        out["CellBalance"] = (
            out["CellBalance"]
              .map({"우수":"Good","정상":"Normal","경고":"Warning","심각":"Critical"})
              .fillna(out["CellBalance"])
        )

    # 숫자 정리
    if "Price" in out.columns:
        out["Price"] = (out["Price"].astype(str)
                        .str.replace(r"[^\d.\-]", "", regex=True)
                        .pipe(pd.to_numeric, errors="coerce"))
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    if "SoH" in out.columns:
        out["SoH"] = pd.to_numeric(out["SoH"], errors="coerce")
    return out

df = normalize_columns(df_raw)

# 필수/수치 컬럼 확인
if "Model" not in df.columns:
    st.error("엑셀에 '차명/배터리종류/차종/모델' 중 하나가 없어 Model 컬럼을 만들 수 없습니다.")
    st.stop()

num_pool = [c for c in ["Age","SoH","Price"] if c in df.columns]
if len(num_pool) < 2:
    st.error(f"수치 컬럼이 부족합니다(필요≥2). 현재: {num_pool}")
    st.stop()

# ───────────────────── 사이드바 ─────────────────────
models        = sorted(df["Model"].dropna().astype(str).unique())
choice        = st.sidebar.selectbox("차명 선택", models)
show_tsne     = st.sidebar.checkbox("t-SNE 2D 추가", value=True)
show_pca3     = st.sidebar.checkbox("PCA 3D 추가", value=False)
perplexity    = st.sidebar.slider("t-SNE perplexity", 5, 50, 30, 1)
show_profiles = st.sidebar.checkbox("추가 프로파일(가로 스크롤)", value=True)

# 💸 비용 옵션 (최소 과금 구조)
st.sidebar.markdown("### 💸 비용 옵션")
cost_saver = st.sidebar.checkbox("비용 절감 모드(저가 모델·짧은 응답)", value=True)
DEFAULT_MODEL = "gpt-4o-mini"

# Secrets/환경변수에서 키·모델 읽기
_api_key, _model_from_secret = get_openai_conf()
MODEL_NAME = _model_from_secret or DEFAULT_MODEL
MAX_TOKENS = 320 if cost_saver else 600

TEMPERATURE = st.sidebar.slider("요약 temperature", 0.0, 1.0, 0.2, 0.05)
MAX_TOKENS  = 320 if cost_saver else 600

# 상태 표시(선택)
if _api_key:
    st.sidebar.success(f"✅ GPT 사용 가능 (모델: {MODEL_NAME})")
else:
    st.sidebar.warning("🔒 OPENAI_API_KEY 미설정 → 로컬 요약으로 대체")

# ───────────────────── 모델 데이터 준비 ─────────────────────
sub_all = df[df["Model"].astype(str) == str(choice)].copy().dropna(subset=num_pool)
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
if hasattr(X, "toarray"): X = X.toarray()

# ───────────── k = Silhouette + Elbow + Dendrogram → 중앙값 ─────────────
def choose_k_multi(X, ks):
    votes = {}
    # 1) Silhouette
    try:
        sil_scores = [silhouette_score(
            X, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        ) for k in ks if k < len(X)]
        if sil_scores:
            votes["silhouette"] = ks[int(np.argmax(sil_scores))]
    except Exception:
        pass
    # 2) Elbow
    try:
        if _has_yb:
            viz = KElbowVisualizer(KMeans(random_state=42, n_init=10), k=ks, metric="distortion", timings=False)
            viz.fit(X)
            if viz.elbow_value_ is not None:
                votes["elbow"] = int(viz.elbow_value_)
        else:
            inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in ks]
            if len(inertias) >= 2:
                diffs = np.diff(inertias)
                idx = int(np.argmax(diffs))
                votes["elbow"] = ks[idx+1] if idx+1 < len(ks) else ks[-1]
    except Exception:
        pass
    # 3) Dendrogram gap
    try:
        if _has_scipy:
            m = X.shape[0]
            idx = np.arange(m if m <= 200 else 200)  # 앞 200개 샘플 (속도)
            Z = linkage(X[idx], method="ward")
            dists = Z[:,2]; gaps = np.diff(dists)
            if len(gaps) >= 1:
                k_est = m - (int(np.argmax(gaps))+1)
                votes["dendrogram"] = max(2, min(k_est, ks[-1]))
    except Exception:
        pass
    vals = [v for v in [votes.get("silhouette"), votes.get("elbow"), votes.get("dendrogram")] if v is not None]
    return (int(np.median(vals)) if vals else 3), votes

k_final, votes = choose_k_multi(X, ks)
st.caption(f"선택된 k = {k_final} (Sil={votes.get('silhouette','—')}, "
           f"Elbow={votes.get('elbow','—')}, Dend={votes.get('dendrogram','—')} → median)")

# ───────────────────── 학습 & 라벨 ─────────────────────
labels = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit_predict(X)
sub_all = sub_all.copy(); sub_all["cluster"] = labels
clusters = sorted(sub_all["cluster"].unique())

# ── 유틸: fig → png, base64
def fig_to_png(fig, dpi=160):
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()

def to_b64(png_bytes): return base64.b64encode(png_bytes).decode("utf-8")

# ── 공통 CSS(가로 스크롤)
st.markdown("""
<style>
.scroll-x { overflow-x:auto; padding:8px 0 10px; }
.scroll-row { display:inline-flex; gap:16px; }
.scroll-row img { border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); }
.caption-center { text-align:center; color:#6b7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ───────────────────── 결과 그래프(가로 스크롤) ─────────────────────
main_images = []  # (caption, png_bytes)

# PCA 2D
p2 = PCA(2, random_state=42).fit_transform(X)
fig = plt.figure(figsize=(5.2, 4.0))
plt.scatter(p2[:,0], p2[:,1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
plt.title(f"{choice}: PCA 2D (k={k_final})"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
png = fig_to_png(fig); main_images.append(("PCA 2D", png))

# Radar (클러스터 평균, 0~1 정규화)
mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
norm_means = mean_matrix.copy()
for c in num_pool:
    mn, mx = df[c].min(), df[c].max()
    norm_means[c] = 0.5 if (pd.isna(mn) or pd.isna(mx) or mx==mn) else (norm_means[c]-mn)/(mx-mn)

angles = [i/len(num_pool)*2*pi for i in range(len(num_pool))] + [0]
fig = plt.figure(figsize=(5.2, 4.0)); ax = plt.subplot(111, polar=True)
for i in clusters:
    vals = norm_means.loc[i].tolist() + [norm_means.loc[i].tolist()[0]]
    ax.plot(angles, vals, label=f"Cluster {i}"); ax.fill(angles, vals, alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(num_pool)
plt.title(f"{choice}: Radar (k={k_final})"); plt.legend(loc="upper right", bbox_to_anchor=(1.25,1.05))
png = fig_to_png(fig); main_images.append(("Radar", png))

# t-SNE 2D (옵션)
if show_tsne:
    perp = min(perplexity, n-1)
    ts2 = TSNE(n_components=2, perplexity=perp, max_iter=500, random_state=42, init="pca").fit_transform(X)
    fig = plt.figure(figsize=(5.2, 4.0))
    plt.scatter(ts2[:,0], ts2[:,1], c=labels, cmap="tab10", s=55, edgecolors="k", alpha=0.9)
    plt.title(f"{choice}: t-SNE 2D (k={k_final})"); plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2"); plt.tight_layout()
    png = fig_to_png(fig); main_images.append(("t-SNE 2D", png))

# PCA 3D (옵션)
if show_pca3:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    p3 = PCA(3, random_state=42).fit_transform(X)
    fig = plt.figure(figsize=(5.6, 4.2)); ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(p3[:,0], p3[:,1], p3[:,2], c=labels, cmap="tab10", s=50, edgecolors="k", alpha=0.85)
    ax3.set_title(f"{choice}: PCA 3D (k={k_final})"); ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    png = fig_to_png(fig); main_images.append(("PCA 3D", png))

# 화면 표시(가로 스크롤)
html_imgs = "".join([f"<img src='data:image/png;base64,{to_b64(p)}' height='320'/>" for _,p in main_images])
st.markdown(f"<div class='scroll-x'><div class='scroll-row'>{html_imgs}</div></div>", unsafe_allow_html=True)
st.markdown("<div class='caption-center'>좌우 스크롤로 결과 그래프(PCA2D, Radar, 옵션: t-SNE/PCA3D)를 확인하세요.</div>", unsafe_allow_html=True)

# ───────────────────── 추가 프로파일(가로 스크롤) ─────────────────────
profile_images = []
if show_profiles:
    # Boxplots
    for col in num_pool:
        fig = plt.figure(figsize=(6,4)); sns.boxplot(x="cluster", y=col, data=sub_all, palette="tab10")
        plt.title(f"{choice}: {col} by Cluster (k={k_final})")
        profile_images.append((f"Box {col}", fig_to_png(fig)))
    # Count & Stacked
    if "CellBalance" in sub_all.columns:
        fig = plt.figure(figsize=(6,4))
        sns.countplot(x="cluster", hue="CellBalance", data=sub_all, palette="Set2")
        plt.title(f"{choice}: Count of CellBalance by Cluster")
        profile_images.append(("Count CellBalance", fig_to_png(fig)))

        ctab_pct = pd.crosstab(sub_all["cluster"], sub_all["CellBalance"], normalize="index")*100
        ctab_pct = ctab_pct.reindex(clusters, fill_value=0)
        fig = plt.figure(figsize=(6,4)); ax = plt.gca()
        ctab_pct.plot(kind="bar", stacked=True, colormap="Paired", ax=ax)
        plt.title(f"{choice}: CellBalance Distribution (%) by Cluster"); plt.tight_layout()
        profile_images.append(("Stacked CellBalance", fig_to_png(fig)))
    # Heatmap
    mean_matrix = sub_all.groupby("cluster")[num_pool].mean()
    fig = plt.figure(figsize=(6,4))
    sns.heatmap(mean_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{choice}: Numeric Feature Means per Cluster")
    profile_images.append(("Heatmap Means", fig_to_png(fig)))

    html_prof = "".join([f"<img src='data:image/png;base64,{to_b64(p)}' height='300'/>" for _,p in profile_images])
    st.markdown(f"<div class='scroll-x'><div class='scroll-row'>{html_prof}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='caption-center'>추가 프로파일도 가로 스크롤로 확인하세요.</div>", unsafe_allow_html=True)

# ───────────────────── GPT 요약 (Chat Completions) & Word 내보내기 ─────────────────────
st.subheader("🧠 믿:음 분석결과 & Word 내보내기")

# 세션 상태 저장
if "ai_text" not in st.session_state:
    st.session_state.ai_text = None

def summarize_compact(dfm: pd.DataFrame, num_pool: list[str]) -> str:
    """프롬프트용 Compact 통계(입력 토큰 절약)"""
    counts = dfm["cluster"].value_counts().sort_index()
    means  = dfm.groupby("cluster")[num_pool].mean()
    line_counts = f"N={len(dfm)}, k={dfm['cluster'].nunique()}"
    line_means  = " | ".join([
        "C{}: ".format(c) + ", ".join([f"{col} {means.loc[c,col]:.1f}" for col in num_pool])
        for c in counts.index
    ])
    return line_counts + "\n" + line_means

def generate_ai_summary(model: str,
                        k_final: int,
                        votes: dict,
                        dfm: pd.DataFrame,
                        num_pool: list[str],
                        model_name: str,
                        max_tokens: int,
                        temperature: float) -> str:
    stats_compact = summarize_compact(dfm, num_pool)
    try:
        if not _has_openai:
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.")

        # ← 여기! 통합 헬퍼로 가져옴
        api_key, model_from_secret = get_openai_conf()
        # secrets에 모델명이 있으면 덮어씀
        if model_from_secret:
            model_name = model_from_secret
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_msg = (
            "You are a concise Korean data analyst. "
            "군집분석 결과를 250~350자 한국어 본문으로 요약하라. "
            "군집별 (연식·SoH·가격) 비교와 실무 활용 포인트 2~3개 포함. "
            "불필요한 표/이모지/목록은 지양."
        )
        user_prompt = (
            f"[모델]{model}\n"
            f"[최종 k] {k_final} (Sil={votes.get('silhouette')}, "
            f"Elbow={votes.get('elbow')}, Dend={votes.get('dendrogram')})\n"
            f"[요약통계]\n{stats_compact}"
        )

        resp = client.chat.completions.create(
            model=model_name,            # 예: gpt-4o-mini
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        # 실패 시 로컬 요약(비용 0원)
        cluster_means = dfm.groupby("cluster")[num_pool].mean().round(1)
        top_price = cluster_means["Price"].idxmax() if "Price" in cluster_means.columns else "—"
        return (
            f"[로컬 요약] {model}을(를) k={k_final}로 군집화했습니다. "
            f"SoH·연식·가격 평균 기준 군집 간 차이가 확인됩니다. "
            f"SoH·가격이 높은 군집({top_price})은 리마케팅 타깃, 저SoH 군집은 정밀 점검 권고가 유효합니다."
        )


# ── Word 도구: 한글 폰트 적용
def _apply_korean_fonts(doc, font_name="Malgun Gothic", size_pt=11):
    style = doc.styles["Normal"]
    style.font.name = font_name
    style.font.size = Pt(size_pt)
    rpr = style._element.get_or_add_rPr()
    rFonts = rpr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), font_name)
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    for h in ["Heading 1", "Heading 2", "Heading 3"]:
        if h in doc.styles:
            st_h = doc.styles[h]
            st_h.font.name = font_name
            rpr = st_h._element.get_or_add_rPr()
            rFonts = rpr.get_or_add_rFonts()
            rFonts.set(qn("w:eastAsia"), font_name)
            rFonts.set(qn("w:ascii"), font_name)
            rFonts.set(qn("w:hAnsi"), font_name)

def export_word(
    doc_title: str,
    model: str,
    gpt_analysis_text: str,
    main_imgs: list[tuple[str, bytes]],
    profile_imgs: list[tuple[str, bytes]],
    dfm: pd.DataFrame,
    num_pool: list[str],
    votes: dict,
    k_final: int,
    template_path: Path | None = None,
    font_name: str = "Malgun Gothic"
) -> BytesIO:
    if template_path and template_path.exists():
        doc = Document(str(template_path))
    else:
        doc = Document()
    _apply_korean_fonts(doc, font_name=font_name, size_pt=11)

    # 표지/메타
    doc.add_heading(doc_title, level=0)
    doc.add_paragraph(f"모델: {model}")
    doc.add_paragraph(f"최종 k: {k_final}  (Sil={votes.get('silhouette')}, Elbow={votes.get('elbow')}, Dend={votes.get('dendrogram')})")

    # 분석 결과 (GPT 생성)
    doc.add_heading("분석 결과 (GPT 생성)", level=1)
    for para in gpt_analysis_text.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    # 주요 시각화
    doc.add_heading("주요 시각화", level=1)
    for cap, png in main_imgs:
        doc.add_paragraph(cap)
        doc.add_picture(BytesIO(png), width=Inches(6.2))

    # 클러스터 요약 통계
    doc.add_heading("클러스터 요약 통계", level=1)
    counts = dfm["cluster"].value_counts().sort_index()
    means  = dfm.groupby("cluster")[num_pool].mean().round(2)
    tbl = doc.add_table(rows=1, cols=2 + len(num_pool))
    hdr = tbl.rows[0].cells
    hdr[0].text = "Cluster"; hdr[1].text = "Count"
    for i, c in enumerate(num_pool, start=2): hdr[i].text = f"Mean {c}"
    for c in counts.index:
        row = tbl.add_row().cells
        row[0].text = str(c); row[1].text = str(int(counts[c]))
        for i, col in enumerate(num_pool, start=2): row[i].text = str(means.loc[c, col])

    # 추가 프로파일
    if profile_imgs:
        doc.add_heading("추가 프로파일", level=1)
        for cap, png in profile_imgs:
            doc.add_paragraph(cap)
            doc.add_picture(BytesIO(png), width=Inches(6.2))

    bio = BytesIO(); doc.save(bio); bio.seek(0)
    return bio

# ── UI: 버튼 & 출력
if "ai_text" not in st.session_state:
    st.session_state.ai_text = None

col_a, col_b = st.columns([1,2])
with col_a:
    gen_btn = st.button("🧠 분석결과 생성 & Word로 저장", use_container_width=True)
with col_b:
    if st.session_state.ai_text:
        st.markdown("**🔎 분석 결과 (GPT 생성)**")
        st.write(st.session_state.ai_text)

if gen_btn:
    with st.spinner("GPT 분석결과 생성 및 Word 문서 작성 중..."):
        ai_text = generate_ai_summary(
            model=choice,
            k_final=k_final,
            votes=votes,
            dfm=sub_all,
            num_pool=num_pool,
            model_name=MODEL_NAME,       # 저가 모델
            max_tokens=MAX_TOKENS,       # 응답 상한
            temperature=TEMPERATURE
        )
        st.session_state.ai_text = ai_text

        TEMPLATE_PATH = Path("data/EV_Battery_Report_Full.docx")  # 있으면 스타일 상속, 없으면 무시
        template_to_use = TEMPLATE_PATH if TEMPLATE_PATH.exists() else None

        word_buf = export_word_like_full(
    doc_title=f"EV 배터리 군집 분석 보고서 – {choice}",
    model=choice,
    gpt_analysis_text=ai_text,                  # GPT 요약(한글)
    main_imgs=main_images,
    profile_imgs=profile_images if show_profiles else [],
    dfm=sub_all,
    num_pool=num_pool,
    votes=votes,
    k_final=k_final,
    font_name="Malgun Gothic"                   # 한글 폰트
)

    st.success("보고서를 생성했습니다.")
    st.download_button(
        "⬇️ Word 파일 다운로드",
        data=word_buf,
        file_name=f"EV_Battery_Report_{choice}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )

import os
from openai import OpenAI
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if api_key:
    try:
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content":"한 줄로 대답: 연결 테스트 성공 여부만 말해줘."}],
            max_tokens=10,
            temperature=0.0,
        )
        st.success("응답: " + r.choices[0].message.content)
    except Exception as e:
        st.error(f"API 호출 실패: {e}")
