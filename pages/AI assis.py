# -*- coding: utf-8 -*-

# ───────────────────── 사이드바(공통) ─────────────────────
with st.sidebar:
    # 브랜드 영역
    st.markdown(
        '<div style="position:sticky;top:0;z-index:10;background:#0f1b2d;padding:12px 12px 6px;'
        'margin:0 -8px 8px -8px;border-bottom:1px solid rgba(255,255,255,.06);">'
        '<div style="font-weight:900;font-size:24px;letter-spacing:.8px;color:#fff;line-height:1.2;">BATTERY-INFO</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # 1) 메인 화면
    st.page_link(home, label="메인 화면", icon="🏠")

    # 간격
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # 2) 분석 결과 확인(군집/기업/시세만)
    with st.expander("분석 결과 확인", expanded=True):
        st.page_link(pg_kmeans, label="군집 분석", icon="🚗")
        st.page_link(pg_reco,   label="기업 추천", icon="✨")
        st.page_link(pg_ts,     label="시세 분석", icon="📈")

    # 3) 이상거래 의심 — 그룹 밖에 단독 배치
    st.page_link(pg_fraud, label="이상거래 의심", icon="🌳")
    st.page_link(pg_ocr,   label="OCR", icon="📄")
    st.page_link(pg_ai_assis, label="AI 정책지원비서", icon="🤖")

st.markdown(
    """
    <style>
      /* ...여기 기존 스타일들... */

      /* 사이드바 page_link(버튼) 텍스트 보이게 */
      section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
      section[data-testid="stSidebar"] [data-testid^="stPageLink"] span {
        color:#EAF2FF !important;  /* 글자색 밝게 */
        opacity:1 !important;
      }
      /* 선택된 페이지(현재 페이지)도 가독성 유지 */
      section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-current="page"] span {
        color:#FFFFFF !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
