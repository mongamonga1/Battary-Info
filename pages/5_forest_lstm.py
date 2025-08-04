# -*- coding: utf-8 -*-
"""Forest LSTM (Streamlit 페이지용 · Colab 매직 제거)"""

import os, re, io, sys, math, json, textwrap, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from pathlib import Path                       # 🔸 추가

# PyTorch (LSTM용) 사용 가능 여부
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
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

# 🔸 Colab 경로 → 프로젝트 내부 상대 경로
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
    df = df.copy()
    grp = df.groupby(group_cols, group_keys=False)
    df[f"{prefix}_mean"] = grp[target].apply(lambda s: s.shift().expanding().mean())
    df[f"{prefix}_std"]  = grp[target].apply(lambda s: s.shift().expanding().std())
    df[f"{prefix}_cnt"]  = grp[target].apply(lambda s: s.shift().expanding().count())
    return df

def z_from(val, mean, std):
    std = std.replace(0, np.nan)
    return (val - mean) / std

def prior_partner_count(df_in):
    df_in = df_in.sort_values("계약일")
    out = pd.Series(index=df_in.index, dtype=float)
    for _, g in df_in.groupby("판매업체"):
        seen = set()
        counts = []
        for _, row in g.iterrows():
            counts.append(len(seen))
            seen.add(row["구매업체"])
        out.loc[g.index] = counts
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
    return np.stack([arr[i:i+seq_len] for i in range(len(arr)-seq_len+1)])

# ----------------------------- 4) CSV 로드 -----------------------------
DATA_PATH = Path("data/통합거래내역.csv")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"읽은 파일: {DATA_PATH.name}, shape={df.shape}")

# ----------------------------- 5) 정제 -----------------------------
for col in ["개당가격", "총계약금액", "계약보증금"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                  .str.replace(r"[^\d\-]", "", regex=True)
                  .replace("", np.nan)
                  .astype(float)
        )

money_cols = ["개당 가격", "총계약금액", "계약보증금"]
for c in money_cols:
    if c in df.columns:
        df[c] = df[c].apply(parse_money)

date_cols = ["계약일", "총지급일"]
for c in date_cols:
    if c in df.columns:
        df[c] = df[c].apply(to_datetime)

df["지급지연일수"] = (df["총지급일"] - df["계약일"]).dt.days
df["보증금율"] = df["계약보증금"] / df["총계약금액"]
df["총계약단가"] = df["총계약금액"] / df["계약수량(단위당)"].replace(0, np.nan)

df = df.sort_values("계약일").reset_index(drop=True)
train_mask = df["계약일"] <= CUTOFF_DATE
df_train = df[train_mask].copy()

# ----------------------------- 6) 피처링 -----------------------------
df_feat = df.copy()
cat_cols = ["판매업체", "구매업체", "제품구분", "배터리종류", "지급형태"]
for c in cat_cols:
    freq = df_train[c].value_counts()
    df_feat[f"{c}_freq"] = df_feat[c].map(freq).fillna(0)

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

for tcol, pfx in [
    ("개당가격",        "판매업체_unit_price"),
    ("총계약금액",      "판매업체_total_amount"),
    ("보증금율",       "판매업체_deposit_rate"),
    ("지급지연일수",    "판매업체_delay_days"),
    ("계약수량(단위당)", "판매업체_qty"),
]:
    if tcol in df_feat.columns:
        df_feat = add_expanding_stats(df_feat, ["판매업체"], tcol, pfx)

df_feat["z_unit_price"]   = z_from(df_feat["개당가격"], df_feat["판매업체_unit_price_mean"], df_feat["판매업체_unit_price_std"])
df_feat["z_total_amount"] = z_from(df_feat["총계약금액"], df_feat["판매업체_total_amount_mean"], df_feat["판매업체_total_amount_std"])
df_feat["z_deposit_rate"] = z_from(df_feat["보증금율"], df_feat["판매업체_deposit_rate_mean"], df_feat["판매업체_deposit_rate_std"])
df_feat["z_delay_days"]   = z_from(df_feat["지급지연일수"], df_feat["판매업체_delay_days_mean"], df_feat["판매업체_delay_days_std"])
df_feat["z_qty"]          = z_from(df_feat["계약수량(단위당)"], df_feat["판매업체_qty_mean"], df_feat["판매업체_qty_std"])

df_feat["판매업체_prior_partner_cnt"] = prior_partner_count(df_feat[["판매업체","구매업체","계약일"]].copy())

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
    n_estimators=500,
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

        Xtr = torch.tensor(train_seqs, dtype=torch.float32).to(device)
        Xall = torch.tensor(all_seqs, dtype=torch.float32).to(device)

        model = _LSTMAE(len(lstm_feature_cols)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(LSTM_EPOCHS):
            idx = torch.randperm(Xtr.size(0))
            for i in range(0, Xtr.size(0), LSTM_BATCH):
                sel = idx[i:i+LSTM_BATCH]
                batch = Xtr[sel]
                opt.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = ((model(Xtr) - Xtr) ** 2).mean(dim=(1,2)).cpu().numpy()
            all_loss = ((model(Xall) - Xall) ** 2).mean(dim=(1,2)).cpu().numpy()

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
def combine_scores(a, b):
    if pd.isna(b):
        return a
    return max(0.6*a, 0.4*b)

df_feat["final_score"] = [combine_scores(a, b) for a, b in zip(df_feat["if_score_norm"], df_feat["lstm_score_norm"])]

import streamlit as st
import plotly.express as px

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
cols = ["계약번호", "계약일", "판매업체", "구매업체", "제품구분", "배터리종류", "final_score"]
top_anom = anom_df[cols].sort_values("final_score", ascending=False)

st.subheader(f"🚨 Top {int(CONTAMINATION*100)}% 이상치 리스트 ({len(top_anom)}건)")
st.dataframe(top_anom, use_container_width=True)
