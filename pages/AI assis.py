# pages/AI assis.py
import streamlit as st

# ── 다른 페이지와 동일한 사이드바/배경 테마 (이 파일에서만 주입)
st.markdown("""
<style>
  /* 배경/헤더 */
  [data-testid="stAppViewContainer"]{background:#f6f8fb;}
  [data-testid="stHeader"]{background:rgba(246,248,251,.7); backdrop-filter:blur(6px);}

  /* 사이드바 다크 */
  section[data-testid="stSidebar"]{background:#0f1b2d; color:#d7e1f2;}
  section[data-testid="stSidebar"] * {font-weight:500;}

  /* page_link / 링크 버튼 공통 */
  section[data-testid="stSidebar"] a[href]{
    color:#EAF2FF !important; opacity:1 !important;
    display:block; padding:10px 12px; border-radius:10px; font-weight:700; text-decoration:none;
  }
  section[data-testid="stSidebar"] a[href]:hover{
    background:#13233b !important; color:#ffffff !important;
  }
  section[data-testid="stSidebar"] a[aria-current="page"]{
    background:#1c2e4a !important; color:#ffffff !important; box-shadow: inset 0 0 0 1px #273b5c;
  }
  /* page_link 내부 글자색 고정 */
  section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] span,
  section[data-testid="stSidebar"] [data-testid^="stPageLink"] span { color:#EAF2FF !important; opacity:1 !important; }
  section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-current="page"] span { color:#FFFFFF !important; }

  /* 사이드바 Expander(분석 결과 확인) 스타일 */
  section[data-testid="stSidebar"] details{
    border:1px solid rgba(255,255,255,.08); border-radius:12px; overflow:hidden; margin:6px 0 12px;
    background:rgba(255,255,255,.04);
  }
  section[data-testid="stSidebar"] details > summary{
    list-style:none; cursor:pointer; padding:10px 12px; color:#EAF2FF; background:rgba(255,255,255,.10);
  }
  section[data-testid="stSidebar"] details[open] > summary{ background:rgba(255,255,255,.16); }
  section[data-testid="stSidebar"] details a[href]{ margin:4px 6px; }
</style>
""", unsafe_allow_html=True)


def render_ai_secretary():
    st.title("🤖 AI 정책지원비서")
    st.caption("프로토타입: 실제 웹연동/LLM모델 분석은 미구현")

    # ── 질문 입력(한 번만 생성) ──
    st.text_area(
        "궁금한 점을 적어주세요",
        key="user_query",
        value=st.session_state.get("user_query", ""),
        placeholder="예) 전기차 배터리 재활용 의무비율을 어떻게 설정해야 하나요?"
    )

    # ── 빠른 질문 버튼(중복 금지 & 고유 key) ──
    st.markdown("**빠른 질문**")

    def set_query(q: str):
        st.session_state["user_query"] = q

    qcols = st.columns(6)
    quicks = [
        "배터리 재활용 의무비율 몇 %가 적절할까?",
        "전기차 보조금 구조 개편 방향은?",
        "ESS 안전기준 강화의 비용·효과는?",
        "배터리 국산화율 제고 방안은?",
        "탄소국경조정제도 대응 전략은?",
        "중고 배터리 거래 투명성 제고?"
    ]
    for i, q in enumerate(quicks):
        qcols[i % 6].button(f"#{i+1}", key=f"quick_{i}", on_click=set_query, args=(q,))

    st.markdown("<div class='blank'></div>", unsafe_allow_html=True)

    # ── 옵션들 ──
    opt1, opt2, opt3 = st.columns([1.2, 1.2, 1])
    with opt1:
        mode = st.radio("분석 모드", ["전체", "요약", "리스크", "대안"], key="mode", horizontal=True)
    with opt2:
        sources = st.multiselect(
            "데이터 소스(가정)",
            ["시세 데이터", "이상거래 탐지", "군집분석", "OCR 문서", "외부(통계/특허/논문)"],
            default=["시세 데이터", "군집분석", "외부(통계/특허/논문)"],
            key="sources"
        )
    with opt3:
        depth = st.slider("근거 강도", 1, 5, 3, key="depth")

    # ── 실행/초기화 ──
    def clear_query():
        st.session_state["user_query"] = ""

    c1, c2, _ = st.columns([1, 1, 5])
    run = c1.button("🔎 분석 실행", key="run_btn")
    c2.button("🧹 초기화", key="clear_btn", on_click=clear_query)

    # ── 결과 패널 ──
    query = st.session_state.get("user_query", "")
    if run and query.strip():
        import hashlib, random
        seed = int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rng = random.Random(seed)

        market = rng.randint(60, 92)
        tech   = rng.randint(55, 90)
        reg    = rng.randint(50, 88)
        cost_down = rng.uniform(4, 17)
        jobs  = rng.randint(1200, 5200)
        co2   = rng.uniform(0.18, 0.85)
        conf  = rng.randint(68, 92)
        option = rng.choice(["단계적 도입", "시범사업 후 의무화", "보조금+규제 병행"])

        st.subheader("🧾 정책 제안서(프로토타입)")
        st.markdown(f"**정책의 근거자료로서**, 아래와 같이 판단합니다. *(내부 신뢰도 추정치: {conf}%)*")

        st.markdown("### 🧩 핵심 결론")
        st.write(f"- 제안: **{option}**")
        st.write(f"- 기대효과): 원가 **{cost_down:.1f}%** 절감 · 신규 일자리 **{jobs:,}개** · CO₂ **{co2:.2f} Mt** 감축/년")

        st.markdown("### 📊 정량 근거")
        st.write(f"- 사용한 소스(가정): {', '.join(sources) if sources else '선택 안 함'}")
        st.write(f"- 시장성 **{market}**, 기술성 **{tech}**, 규제 적합성 **{reg}** (0~100 가중지수)")

        if mode in ("전체", "요약"):
            st.markdown("### 🔎 정성 근거(요약)")
            st.write("- 해외 동향: 미국·EU는 인센티브와 의무비율 병행 추세")
            st.write("- 산업 파급: 회수/재제조 생태계 활성화 및 중소협력사 역량 강화")

        if mode in ("전체", "리스크"):
            st.markdown("### ⚠️ 리스크 · 한계")
            st.write("- 단기 비용 증가와 데이터 표준 부재 → **표준화·인증 가이드라인** 필요")
            st.write("- 보조금 집중 시 시장왜곡 가능 → **성과연동·감액 장치** 병행")

        if mode in ("전체", "대안"):
            st.markdown("### 🧭 정책 옵션(택1 또는 병행)")
            st.write("1) 시범사업(1~2년) 후 의무비율 단계 상향")
            st.write("2) 성과연동 보조금(효율·회수율 기준 차등)")
            st.write("3) 공공조달 가점 및 민관 표준 데이터셋 구축")

        st.caption("※ 모든 수치/근거는 데모용 모의 값입니다. 실제 분석엔 내부/외부 데이터 파이프라인과 모델을 연결하세요.")
    else:
        st.info("왼쪽의 **빠른 질문 버튼**을 누르거나 텍스트를 작성한 뒤 **분석 실행**을 클릭하세요.")


# 함수 호출
render_ai_secretary()
