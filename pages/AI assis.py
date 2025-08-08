# pages/ai_secretary.py  (예시)
import streamlit as st

st.set_page_config(
    page_title="AI 정책지원비서",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 👉 홈과 동일한 사이드바/배경 스타일 주입
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f6f8fb; }
  [data-testid="stHeader"] { background: rgba(246,248,251,0.7); backdrop-filter: blur(6px); }
  [data-testid="stSidebar"] { background: #0f1b2d; color: #d7e1f2; }
  [data-testid="stSidebar"] * { font-weight: 500; }

  /* 사이드바 링크 스타일 (홈과 동일) */
  [data-testid="stSidebar"] a[href]{
    color:#EAF2FF !important; opacity:1 !important;
    display:block; padding:10px 12px; border-radius:10px; font-weight:700;
  }
  [data-testid="stSidebar"] a[href]:hover{
    background:#13233b !important; color:#ffffff !important;
  }
  [data-testid="stSidebar"] a[aria-current="page"]{
    background:#1c2e4a !important; color:#ffffff !important;
    box-shadow: inset 0 0 0 1px #273b5c;
  }

  /* page_link 글자색 고정 (메인 파일 하단 CSS와 동일) */
  section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
  section[data-testid="stSidebar"] [data-testid^="stPageLink"] span {
    color:#EAF2FF !important; opacity:1 !important;
  }
  section[data-testid="stSidebar"]
  [data-testid="stBaseButton-secondary"][aria-current="page"] span {
    color:#FFFFFF !important;
  }
</style>
""", unsafe_allow_html=True)

# --- 페이지 내용(빈 상태 OK) ---
st.title("🤖 AI 정책지원비서")
st.info("페이지 전환만 구현되어 있습니다. (기능은 추후 연결)")
