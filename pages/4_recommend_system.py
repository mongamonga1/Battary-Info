# -*- coding: utf-8 -*-
"""
Recommend system - 사용자 선택 기업 기반 추천
"""
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
# ------------------------------------------------------------------
# 1) 데이터 불러오기
# ------------------------------------------------------------------
TX_PATH     = "data/통합거래내역.csv"
MARKET_PATH = "data/매물리스트.csv"
EVAL_PATH   = "data/업체평가정보.csv"

transactions = pd.read_csv(TX_PATH)
market       = pd.read_csv(MARKET_PATH)
eval_df      = pd.read_csv(EVAL_PATH)

# ------------------------------------------------------------------
# 2) 대상 기업 선택 (사이드바)
# ------------------------------------------------------------------
company_list = sorted(pd.unique(
    pd.concat([
        transactions["판매업체"],
        transactions["구매업체"]
    ])
))
default_idx = company_list.index("성일하이텍(주)") if "성일하이텍(주)" in company_list else 0
target = st.sidebar.selectbox("🔍 기업 선택", company_list, index=default_idx)

# ------------------------------------------------------------------
# 3) 판매자+구매자 거래내역 통합
# ------------------------------------------------------------------
all_tx = pd.concat([
    transactions[["판매업체","제품구분"]].rename(columns={"판매업체":"업체명"}),
    transactions[["구매업체","제품구분"]].rename(columns={"구매업체":"업체명"})
])

# ------------------------------------------------------------------
# 4) 기업 × 제품구분 매트릭스 생성
# ------------------------------------------------------------------
cust_product_matrix = all_tx.pivot_table(
    index="업체명", columns="제품구분", aggfunc=len, fill_value=0
)

# ------------------------------------------------------------------
# 5) 유사기업 Top-5 선정 (Cosine Similarity)
# ------------------------------------------------------------------
cos_sim    = cosine_similarity(cust_product_matrix)
cos_sim_df = pd.DataFrame(cos_sim,
                          index=cust_product_matrix.index,
                          columns=cust_product_matrix.index)

similar_customers = (
    cos_sim_df[target]
    .sort_values(ascending=False)
    .drop(index=target)
    .head(5)
)

# ------------------------------------------------------------------
# 6) 추천 점수 합산
# ------------------------------------------------------------------
# 6-1) 유사기업 거래비중
similar_tx = all_tx[all_tx["업체명"].isin(similar_customers.index)]
sim_counts = similar_tx["제품구분"].value_counts(normalize=True).to_dict()

# 6-2) 내 거래비중
my_tx     = all_tx[all_tx["업체명"] == target]
my_counts = my_tx["제품구분"].value_counts(normalize=True).to_dict()

# 6-3) 하이브리드 점수 계산
product_scores = {}
for item in set(sim_counts) | set(my_counts):
    product_scores[item] = 0.5 * sim_counts.get(item, 0) + 0.5 * my_counts.get(item, 0)

top_items = sorted(product_scores, key=product_scores.get, reverse=True)

# ------------------------------------------------------------------
# 7) 매물 필터링 + 추가 정보
# ------------------------------------------------------------------
filtered_market = market[market["제품구분"].isin(top_items)].copy()

# 이전 거래업체 여부
tx_cust      = transactions.query("판매업체==@target or 구매업체==@target")
prev_partners = set(tx_cust["판매업체"]) | set(tx_cust["구매업체"]) - {target}
filtered_market["이전거래업체"] = filtered_market["판매업체"].isin(prev_partners)

# 제품구분 우선순위
filtered_market["제품구분우선순위"] = filtered_market["제품구분"].apply(
    lambda x: top_items.index(x) if x in top_items else len(top_items)
)

# 총평가점수 병합
filtered_market = filtered_market.merge(
    eval_df[["업체명","총평가점수"]],
    left_on="판매업체", right_on="업체명", how="left"
)

# ------------------------------------------------------------------
# 8) 정렬
# ------------------------------------------------------------------
filtered_market = filtered_market.sort_values(
    by=["제품구분우선순위","이전거래업체","총평가점수"],
    ascending=[True, False, False]
)

# ------------------------------------------------------------------
# 9) 결과 출력: 선택 기업을 위한 추천 매물 Top 10
# ------------------------------------------------------------------
columns = ["판매업체","제품구분","배터리종류","계약수량(단위당)","총평가점수"]
result  = filtered_market.head(10)[columns].reset_index(drop=True)

st.subheader(f"📊 {target} 추천 매물 Top 10")
st.dataframe(result, use_container_width=True)
