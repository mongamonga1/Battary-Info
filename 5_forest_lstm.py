# -*- coding: utf-8 -*-
"""Forest LSTM (Streamlit 페이지용 · 경량화/안정화 패치, 모델 파라미터 불변)"""

import os, re, io, sys, math, json, textwrap, warnings
warnings.filterwarnings("ignore")

# ⬇️ 소형 CPU에서 과도한 스레딩 방지(속도/안정성 ↑)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pathlib import Path
import streamlit as st
import plotly.express as px

# PyTorch (LSTM용) 사용 가능 여부
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
    # ⬇️ CPU 스레드 1개로 고정(작은 머신에서 컨텍스트 스위칭 ↓)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception:
    TORCH_AVAILABLE = False

# ── 경량 테마(색상) ──────────────────────────────────────────────
def apply_colors(page_bg="#F5F7FB", sidebar_bg="#0F172A", sidebar_text="#DBE4FF", sidebar_link="#93C5FD"):
    st.markdown(f"""
    <style>
      .stApp {{ background: {page_bg}; }}
      section[data-testid="stSidebar"] {{ background: {sidebar_bg}; }}
      section[data-testid="stSidebar"] * {{ color: {sidebar_text} !important; }}
      section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] svg {{
        color: {sidebar_link} !important; fill: {sidebar_link} !important;
      }}
      section[data-testid="stSidebar"] a:hover {{ background-color: rgba(255,255,255,0.08) !important; border-radius: 8px; }}
    </style>
    """, unsafe_allow_html=True)

apply_colors(
    page_bg="#F5F7FB",
    sidebar_bg="#0F172A",
    sidebar_text="#FFFFFF",
    sidebar_link="#93C5FD"
)
st.markdown("""
<style>
/* (1) 드롭존 박스 */
section[data-testid="stSidebar"] [data-testid*="FileUploaderDropzone"]{
  background-color:#1E293B !important;
  border:1.5px dashed #94A3B8 !important;
  border-radius:12px !important;
}
/* (2) 호환용(기존 클래스 경로) */
section[data-testid="stSidebar"] .stFileUploader [data-testid*="FileUploaderDropzone"],
section[data-testid="stSidebar"] .stFileUploader > div > div{
  background-color:#1E293B !important;
  border:1.5px dashed #94A3B8 !important;
  border-radius:12px !important;
}
/* (3) 드롭존 내부 안내문 텍스트만 밝게 — '버튼'은 제외 */
section[data-testid="stSidebar"] [data-testid*="FileUploaderDropzone"] *:not(button):not([role="button"]):not(button *):not([role="button"] *),
section[data-testid="stSidebar"] .stFileUploader [data-testid*="FileUploaderDropzone"] *:not(button):not([role="button"]):not(button *):not([role="button"] *){
  color:#EAF2FF !important;
  opacity:1 !important;
  filter:none !important;
}
/* (4) 업로더의 ‘Browse files’ 버튼(및 라벨)만 진하게 */
section[data-testid="stSidebar"] [data-testid*="FileUploader"] button,
section[data-testid="stSidebar"] [data-testid*="FileUploader"] [role="button"],
section[data-testid="stSidebar"] [data-testid*="FileUploader"] button *,
section[data-testid="stSidebar"] [data-testid*="FileUploader"] [role="button"] *{
  background-color:#F1F5F9 !important;
  color:#0F172A !important;
  font-weight:700 !important;
  opacity:1 !important;
}
/* 사이드바 selectbox(입력창) 텍스트만 검정 */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *{
  color:#0F172A !important;
}
/* (옵션) 펼쳐진 옵션 목록 텍스트도 검정 */
div[data-baseweb="popover"] [data-baseweb="menu"] *{
  color:#0F172A !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- 1) 설정값 -----------------------------
SEED = 42
np.random.seed(SEED)

CUTOFF_DATE = pd.Timestamp("2025-07-27")
CONTAMINATION = 0.02
RUN_LSTM = True
MIN_TXN_FOR_LSTM = 40
SEQ_LEN = 12
LSTM_EPOCHS = 40
LSTM_LR = 1e-3
LSTM_BATCH = 64
LSTM_HIDDEN = 16
LSTM_LAYERS = 1

# 🔸 프로젝트 상대 경로
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------- 2) 유틸 함수 -----------------------------
def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^\d\-]", "", str(x))
    return float(s) if s else np.nan

def to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def minmax(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def add_expanding_stats(df, group_cols, target, prefix):
    """
    벡터화된 '과거까지의' 누적 통계:
      - mean/std/count (현재 row는 제외: shift(1) 효과)
    Python 루프 없이 groupby + cumsum으로 계산해 속도 개선.
    """
    out = df.copy()
    # 숫자형으로 캐스팅
    x = out[target].astype(float)
    # 현재행 제외(사전 정보만 사용)
    # x_shift는 out[target]을 group별로 한 칸 밀어낸 값
    x_shift = out.groupby(group_cols)[target].shift(1).astype(float)

    # 누적합(sum1), 제곱 누적합(sum2), 카운트(n)
    tmp = out.copy()
    tmp["_xs"] = x_shift.fillna(0.0)
    tmp["_xs2"] = (x_shift ** 2).fillna(0.0)
    sum1 = tmp.groupby(group_cols)["_xs"].cumsum()
    sum2 = tmp.groupby(group_cols)["_xs2"].cumsum()
    n = tmp.groupby(group_cols).cumcount()  # 과거 건수(현재 제외)

    # 평균/표준편차 계산(샘플 표준편차, n-1 분모)
    mean = sum1 / n.replace(0, np.nan)
    var = (sum2 - (sum1 ** 2) / n.replace(0, np.nan)) / (n - 1).replace({0: np.nan})
    std = np.sqrt(var)

    out[f"{prefix}_mean"] = mean.values
    out[f"{prefix}_std"] = std.values
    out[f"{prefix}_cnt"] = n.values.astype(float)

    # 클린업
    out.drop(columns=[c for c in ["_xs", "_xs2"] if c in out.columns], inplace=True, errors="ignore")
    return out

def z_from(val, mean, std):
    std = std.replace(0, np.nan)
    return (val - mean) / std

def prior_partner_count(df_in):
    """
    판매업체별로 계약일 오름차순 정렬 시,
    해당 시점 이전까지 고유 구매업체 수(벡터화).
    """
    d = df_in[["판매업체", "구매업체", "계약일"]].copy()
    d = d.sort_values(["판매업체", "계약일", "구매업체"])
    # (판매업체, 구매업체) 쌍의 첫 등장 표시
    first_pair = ~d[["판매업체", "구매업체"]].duplicated(keep="first")
    # 업체별 고유 구매업체 누적 수
    cum_unique = first_pair.groupby(d["판매업체"]).cumsum()
    # 현재 행 이전의 고유 수 = 누적 - (현재가 첫 등장인지)
    prior = (cum_unique - first_pair.astype(int)).astype(float)
    out = pd.Series(index=df_in.index, dtype=float)
    out.loc[d.index] = prior.values
    return out

# ----------------------------- 3) LSTM 오토인코더 정의 -----------------------------
class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_size=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_dim, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        z, _ = self.encoder(x)
        last = z[:, -1:, :]
        last_rep = last.repeat(1, x.size(1), 1)
        out, _ = self.decoder(last_rep)
        return out

def make_sequences(arr, seq_len):
    # (N, F) → (N-seq_len+1, seq_len, F)
    if len(arr) < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=arr.dtype)
    # 벡터화된 슬라이싱
    idx = np.arange(seq_len)[None, :] + np.arange(len(arr) - seq_len + 1)[:, None]
    return arr[idx]

# ----------------------------- 4) CSV 로드 (캐시 + PyArrow 안전 폴백) -----------------------------
DATA_PATH = Path("data/통합거래내역.csv")

def _file_sig(path: Path):
    try:
        stt = path.stat()
        return (str(path), stt.st_size, int(stt.st_mtime))
    except FileNotFoundError:
        return (str(path), None, None)

@st.cache_data(show_spinner=False)
def read_csv_fast(path: Path, encoding="utf-8-sig", arrow=True, memory_map=True, low_memory=False):
    """
    - 먼저 pyarrow 엔진 시도(가능하면 가장 빠름)
      · 이때는 pandas 버전에 따라 지원 인자가 다르므로 최소 인자만 전달
      · dtype_backend='pyarrow'도 pandas 2.x에서만 가능 → 실패 시 재시도
    - 실패하면 pandas 기본 엔진으로 폴백
    """
    _ = _file_sig(path)  # 파일 변경 시 캐시 무효화 키

    # 1) pyarrow 엔진 우선 시도 (있을 때만)
    if arrow:
        try:
            import pyarrow  # noqa: F401
            # (a) pandas 2.x에서만 동작하는 경로
            try:
                return pd.read_csv(path, engine="pyarrow", encoding=encoding, dtype_backend="pyarrow")
            except TypeError:
                # (b) dtype_backend 미지원 → 최소 인자만
                return pd.read_csv(path, engine="pyarrow", encoding=encoding)
        except Exception:
            # pyarrow 미설치/미지원 → 아래 기본 엔진 폴백
            pass

    # 2) 기본 엔진 폴백
    try:
        # pandas C 엔진에서는 memory_map/low_memory가 유효
        return pd.read_csv(path, encoding=encoding, memory_map=memory_map, low_memory=low_memory)
    except TypeError:
        # 환경에 따라 일부 인자 미지원 시 최소 인자로 재시도
        return pd.read_csv(path, encoding=encoding)

df = read_csv_fast(DATA_PATH)
# st.write(f"읽은 파일: {DATA_PATH.name}, shape={df.shape}")

# ----------------------------- 5) 정제 -----------------------------
for col in ["개당가격", "총계약금액", "계약보증금"]:
    if col in df.columns:
        # 불필요 문자 제거 → float
        s = (df[col].astype(str)
                 .str.replace(r"[^\d\-]", "", regex=True)
                 .replace("", np.nan))
        df[col] = pd.to_numeric(s, errors="coerce")

money_cols = ["개당 가격", "총계약금액", "계약보증금"]
for c in money_cols:
    if c in df.columns:
        df[c] = df[c].apply(parse_money)

date_cols = ["계약일", "총지급일"]
for c in date_cols:
    if c in df.columns:
        df[c] = df[c].apply(to_datetime)

if {"총지급일","계약일"}.issubset(df.columns):
    df["지급지연일수"] = (df["총지급일"] - df["계약일"]).dt.days
if {"계약보증금","총계약금액"}.issubset(df.columns):
    df["보증금율"] = df["계약보증금"] / df["총계약금액"]
if {"총계약금액","계약수량(단위당)"}.issubset(df.columns):
    df["총계약단가"] = df["총계약금액"] / df["계약수량(단위당)"].replace(0, np.nan)

df = df.sort_values("계약일").reset_index(drop=True)
train_mask = df["계약일"] <= CUTOFF_DATE
df_train = df[train_mask].copy()

# ----------------------------- 6) 피처링 -----------------------------
df_feat = df.copy()
df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

# 범주 빈도(훈련기간 기준) → 전체에 매핑
cat_cols = ["판매업체", "구매업체", "제품구분", "배터리종류", "지급형태"]
for c in cat_cols:
    if c in df.columns:
        freq = df_train[c].value_counts()
        df_feat[f"{c}_freq"] = df_feat[c].map(freq).fillna(0)

# 일부 데이터셋에서 '지급 형태'로 표기된 경우 호환
if "지급형태_freq" in df_feat.columns and "지급 형태_freq" not in df_feat.columns:
    df_feat["지급 형태_freq"] = df_feat["지급형태_freq"]

df_feat["key_판매구매"] = df_feat["판매업체"].astype(str) + "∥" + df_feat["구매업체"].astype(str)
df_feat["key_판매제품배터리"] = (
    df_feat["판매업체"].astype(str) + "∥" +
    df_feat["제품구분"].astype(str) + "∥" +
    df_feat["배터리종류"].astype(str)
)

seen_판매구매 = set(df_train["판매업체"].astype(str) + "∥" + df_train["구매업체"].astype(str))
seen_판매제품배터리 = set(
    df_train["판매업체"].astype(str) + "∥" +
    df_train["제품구분"].astype(str) + "∥" +
    df_train["배터리종류"].astype(str)
)
df_feat["신규_판매구매"] = (~df_feat["key_판매구매"].isin(seen_판매구매)).astype(int)
df_feat["신규_판매제품배터리"] = (~df_feat["key_판매제품배터리"].isin(seen_판매제품배터리)).astype(int)

# 판매업체 기준 과거 누적 통계(벡터화 버전)
for tcol, pfx in [
    ("개당가격",        "판매업체_unit_price"),
    ("총계약금액",      "판매업체_total_amount"),
    ("보증금율",        "판매업체_deposit_rate"),
    ("지급지연일수",     "판매업체_delay_days"),
    ("계약수량(단위당)", "판매업체_qty"),
]:
    if tcol in df_feat.columns:
        df_feat = add_expanding_stats(df_feat, ["판매업체"], tcol, pfx)

# z-score들
def _safe_series(name):
    return df_feat[name] if name in df_feat.columns else pd.Series(np.nan, index=df_feat.index)

df_feat["z_unit_price"]   = z_from(_safe_series("개당가격"),      _safe_series("판매업체_unit_price_mean"),   _safe_series("판매업체_unit_price_std"))
df_feat["z_total_amount"] = z_from(_safe_series("총계약금액"),    _safe_series("판매업체_total_amount_mean"), _safe_series("판매업체_total_amount_std"))
df_feat["z_deposit_rate"] = z_from(_safe_series("보증금율"),      _safe_series("판매업체_deposit_rate_mean"), _safe_series("판매업체_deposit_rate_std"))
df_feat["z_delay_days"]   = z_from(_safe_series("지급지연일수"),  _safe_series("판매업체_delay_days_mean"),   _safe_series("판매업체_delay_days_std"))
df_feat["z_qty"]          = z_from(_safe_series("계약수량(단위당)"), _safe_series("판매업체_qty_mean"),        _safe_series("판매업체_qty_std"))

# 판매업체별 이전 고유 거래처 수(벡터화)
if {"판매업체","구매업체","계약일"}.issubset(df_feat.columns):
    df_feat["판매업체_prior_partner_cnt"] = prior_partner_count(df_feat[["판매업체","구매업체","계약일"]].copy())
else:
    df_feat["판매업체_prior_partner_cnt"] = 0.0

# 피처 목록(원본 유지, 누락분은 0.0으로 채움)
feature_cols = [
    "개당가격","총계약금액","계약보증금","지급지연일수","보증금율","총계약단가","계약수량(단위당)",
    "판매업체_freq","구매업체_freq","제품구분_freq","배터리종류_freq","지급 형태_freq",
    "신규_판매구매","신규_판매제품배터리",
    "z_unit_price","z_total_amount","z_deposit_rate","z_delay_days","z_qty",
    "판매업체_unit_price_cnt","판매업체_total_amount_cnt","판매업체_deposit_rate_cnt","판매업체_delay_days_cnt","판매업체_qty_cnt",
    "판매업체_prior_partner_cnt",
]
for c in feature_cols:
    if c not in df_feat.columns:
        df_feat[c] = 0.0

X_train = df_feat.loc[train_mask, feature_cols].astype(float).fillna(0.0).values
X_all   = df_feat[feature_cols].astype(float).fillna(0.0).values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_all_s   = scaler.transform(X_all)

# ----------------------------- 7) Isolation Forest -----------------------------
if_model = IsolationForest(
    n_estimators=500,            # ← 모델 파라미터 유지
    contamination=CONTAMINATION,
    max_samples="auto",
    bootstrap=True,
    random_state=SEED,
    n_jobs=1
)
if_model.fit(X_train_s)
if_scores = -if_model.decision_function(X_all_s)
df_feat["if_score"] = if_scores
df_feat["if_score_norm"] = minmax(if_scores)

# ----------------------------- 8) (선택) LSTM 보조 스코어 -----------------------------
df_feat["lstm_score_norm"] = np.nan
lstm_rows = []

if RUN_LSTM and TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_feature_cols = ["개당가격","보증금율","지급지연일수","계약수량(단위당)","총계약금액"]

    counts = df_train["판매업체"].value_counts()
    target_vendors = counts[counts >= MIN_TXN_FOR_LSTM].index.tolist()

    class _LSTMAE(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = LSTMAE(input_dim, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS)
        def forward(self, x):
            return self.model(x)

    for vendor in target_vendors:
        g = df_feat[df_feat["판매업체"] == vendor].sort_values("계약일").copy()
        g_train = g[g["계약일"] <= CUTOFF_DATE].copy()
        if g.shape[0] < SEQ_LEN + 1 or g_train.shape[0] < SEQ_LEN + 1:
            continue

        scaler_v = StandardScaler()
        g_train_feat = scaler_v.fit_transform(g_train[lstm_feature_cols].astype(float).fillna(0.0).values)
        g_all_feat   = scaler_v.transform(g[lstm_feature_cols].astype(float).fillna(0.0).values)

        train_seqs = make_sequences(g_train_feat, SEQ_LEN)
        all_seqs   = make_sequences(g_all_feat, SEQ_LEN)
        if train_seqs.shape[0] == 0 or all_seqs.shape[0] == 0:
            continue

        Xtr  = torch.tensor(train_seqs, dtype=torch.float32, device=device)
        Xall = torch.tensor(all_seqs,  dtype=torch.float32, device=device)

        model = _LSTMAE(len(lstm_feature_cols)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(LSTM_EPOCHS):  # ← 에포크 수 유지
            idx = torch.randperm(Xtr.size(0), device=device)
            for i in range(0, Xtr.size(0), LSTM_BATCH):
                sel = idx[i:i+LSTM_BATCH]
                batch = Xtr.index_select(0, sel)
                opt.zero_grad(set_to_none=True)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss  = ((model(Xtr)  - Xtr)  ** 2).mean(dim=(1,2)).detach().cpu().numpy()
            all_loss = ((model(Xall) - Xall) ** 2).mean(dim=(1,2)).detach().cpu().numpy()

        idx_all = g.index.to_list()
        target_row_indices = idx_all[SEQ_LEN-1:]

        p50, p99 = np.percentile(tr_loss, 50), np.percentile(tr_loss, 99)
        denom = max(p99 - p50, 1e-6)
        norm_scores = np.clip((all_loss - p50) / denom, 0, 5) / 5.0

        tmp = pd.DataFrame({
            "판매업체": vendor,
            "row_index": target_row_indices,
            "lstm_loss": all_loss,
            "lstm_score_norm": norm_scores
        })
        lstm_rows.append(tmp)

    if lstm_rows:
        lstm_df = pd.concat(lstm_rows, ignore_index=True)
        df_feat.loc[lstm_df["row_index"].values, "lstm_score_norm"] = lstm_df["lstm_score_norm"].values
else:
    lstm_df = pd.DataFrame(columns=["row_index","lstm_loss","lstm_score_norm","판매업체"])

# ----------------------------- 9) 최종 점수 결합 -----------------------------
if "final_score" in df_feat.columns:
    df_feat = df_feat.drop(columns=["final_score"])
# --------------------------------------------------------------------------
def combine_scores(a, b):
    if pd.isna(b):
        return a
    return max(0.6*a, 0.4*b)

df_feat["final_score"] = [
    combine_scores(a, b) for a, b in zip(df_feat["if_score_norm"], df_feat["lstm_score_norm"])
]
# ───────────────────── 10) Plotly 시각화 ─────────────────────
threshold = df_feat["final_score"].quantile(1 - CONTAMINATION)
df_feat["anomaly"] = df_feat["final_score"] >= threshold

fig_anom = px.scatter(
    df_feat,
    x="계약일", y="final_score",
    color="anomaly",
    color_discrete_map={False: "lightblue", True: "red"},
    title=f"Anomalies (top {int(CONTAMINATION*100)}%) Highlighted",
    labels={"계약일":"Contract Date", "final_score":"Final Score", "anomaly":"Anomaly"},
    hover_data={
        "계약번호": True, "판매업체": True, "구매업체": True,
        "제품구분": True, "배터리종류": True, "final_score": ":.4f"
    }
)
fig_anom.update_traces(marker=dict(size=8))

# Streamlit에 Plotly 차트로 표시
st.subheader("🔍 이상치 스코어 산점도")
st.plotly_chart(fig_anom, use_container_width=True)

# ───────────────────── 11) 이상치 리스트 출력 ─────────────────────
anom_df = df_feat[df_feat["final_score"] >= threshold]
anom_df = anom_df.loc[:, ~anom_df.columns.duplicated()]
cols = ["계약번호", "계약일", "판매업체", "구매업체", "제품구분", "배터리종류", "final_score"]
cols = [c for c in cols if c in anom_df.columns]
top_anom = anom_df[cols].sort_values("final_score", ascending=False)

st.subheader(f"🚨 Top {int(CONTAMINATION*100)}% 이상치 리스트 ({len(top_anom)}건)")
st.dataframe(top_anom, use_container_width=True)
