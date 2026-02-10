import streamlit as st
import requests
import tracka_final as ta
import jeonse_ratio as jr
import numpy as np
import re

st.set_page_config(page_title="전세 위험도", layout="centered")

# === [ADD] 9분면 임계값 ===
# 3단계 분류용 (Safe/Caution/High)
TRACKA_T1 = 0.56  # T1 이상: Caution
TRACKA_T2 = 0.68  # T2 이상: High

TRACKB_T1 = 0.56  # T1 이상: Caution (Track A와 통일)
TRACKB_T2 = 0.68  # T2 이상: High (Track A와 통일)

def go(page_name: str):
    st.session_state.page = page_name
    st.rerun()


def floor_to_num(floor_label: str) -> int:
    s = str(floor_label).strip()

    # "지상 2층", "2층", "지상2층" 전부 대응
    m = re.search(r"(\d+)\s*층", s)
    if m:
        return int(m.group(1))

    # 특수 케이스
    if "반지하" in s:
        return 0      # ← 모델 기준에 맞춰 조정 가능
    if "옥탑" in s:
        return 99     # ← 모델 기준에 맞춰 조정 가능

    return 1

def classify_3bin(prob: float, t1: float, t2: float) -> str:
    """저위험 / 주의 / 고위험"""
    if prob is None:
        return "N/A"
    if prob >= t2:
        return "High"
    if prob >= t1:
        return "Caution"
    return "Safe"


def get_9zone_case(a_level: str, b_level: str):
    """
    9분면 매핑 (3x3)
    Track A (세로): Safe / Caution / High
    Track B (가로): Safe / Caution / High
    """
    zone_map = {
        ("Safe", "Safe"): ("①", "최적 안전존", "사기 패턴과 비슷하지 않고, 시장 급변 시에도 보증금 회수가 확실한 매물입니다.", "#d1fae5", "#00ad00"),
        ("Safe", "Caution"): ("②", "시장 관찰존", "사기 패턴과 떨어져 있으나 집값 하락 시 보증금 일부 손실 가능성이 있습니다.", "#fef3c7", "#f5920b"),
        ("Safe", "High"): ("③", "시장 경고존", "사기 패턴과 떨어져 있으나 시장 붕괴 시 큰 손실이 예상되는 '깡통 전세' 위험입니다.", "#fef3c7", "#f5920b"),
        ("Caution", "Safe"): ("④", "패턴 주의존", "시장은 안정적이나 이전 사기 패턴과 유사점이 포착되었습니다.", "#fef3c7", "#f5920b"),
        ("Caution", "Caution"): ("⑤", "복합 관리존", "시장 위험과 이전 사기 패턴과의 유사도 모두 주의가 필요합니다. 계약 전 전문가 상담을 권장합니다.", "#fef3c7", "#f5920b"),
        ("Caution", "High"): ("⑥", "리스크 심화존", "시장 붕괴와 이전 사기 패턴과의 유사도가 복합된 고위험 상황입니다.", "#fef3c7", "#f5920b"),
        ("High", "Safe"): ("⑦", "사기 경고존", "시장은 좋으나 이전 사기 패턴과의 유사도가 높은 '기획 사기' 의심 매물입니다.", "#fef3c7", "#f5920b"),
        ("High", "Caution"): ("⑧", "위험 확산존", "악의적 사기 설계와 시장 붕괴 위험이 결합된 최악의 시나리오입니다.", "#fef3c7", "#f5920b"),
        ("High", "High"): ("⑨", "절대 금지존", "경매 사고 확률이 압도적으로 높습니다. 어떠한 조건에서도 계약 체결을 권장하지 않습니다.", "#fee2e2", "#dc2626"),
    }
    
    result = zone_map.get((a_level, b_level), ("-", "알 수 없음", "분류 불가", "#f3f4f6", "#6b7280"))
    return result  # (코드, 이름, 설명, 배경색, 텍스트색)


# =========================
# JUSO (도로명주소) 검색 API 설정
# =========================
JUSO_API_URL = "https://business.juso.go.kr/addrlink/addrLinkApi.do"
JUSO_API_KEY = "devU01TX0FVVEgyMDI2MDIwNzE1NDI1MzExNzU3MTY="
JUSO_RESULT_PER_PAGE = 10


@st.cache_data(show_spinner=False, ttl=60)
def juso_search(keyword: str, page: int = 1, count: int = 10):
    params = {
        "confmKey": JUSO_API_KEY,
        "currentPage": str(page),
        "countPerPage": str(count),
        "keyword": keyword,
        "resultType": "json",
    }
    r = requests.get(JUSO_API_URL, params=params, timeout=6)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", {})
    common = results.get("common", {})
    juso_list = results.get("juso", []) or []

    error_code = common.get("errorCode", "-999")
    error_msg = common.get("errorMessage", "알 수 없는 오류")

    total_count = int(common.get("totalCount", "0") or "0")
    current_page = int(common.get("currentPage", page) or page)
    count_per_page = int(common.get("countPerPage", count) or count)

    return {
        "ok": (error_code == "0"),
        "errorCode": error_code,
        "errorMessage": error_msg,
        "totalCount": total_count,
        "currentPage": current_page,
        "countPerPage": count_per_page,
        "juso": juso_list,
    }


# ----------------------------
# State (Router + Inputs)
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

if "inputs" not in st.session_state:
    # ✅ 여기에는 “사용자가 입력한 값”만 저장 (더미 점수 X)
    st.session_state.inputs = {}

# 주소 검색 상태
if "addr_open" not in st.session_state:
    st.session_state.addr_open = False
if "selected_juso" not in st.session_state:
    st.session_state.selected_juso = None
if "addr_query" not in st.session_state:
    st.session_state.addr_query = ""
if "addr_page" not in st.session_state:
    st.session_state.addr_page = 1


def toggle_addr():
    st.session_state.addr_open = not st.session_state.addr_open


def choose_juso(juso_obj: dict):
    st.session_state.selected_juso = juso_obj
    st.session_state.addr_open = False


def parse_contract_years(label: str) -> int:
    # "1년", "2년", "3년", "4년 이상" -> 숫자
    if label.startswith("4"):
        return 4
    try:
        return int(label.replace("년", "").strip())
    except Exception:
        return 2


# ----------------------------
# CSS (FINAL)
# ----------------------------
st.markdown(
    """
    <style>
      /* Adobe Fonts - 210 Supersize */
      @import url("https://use.typekit.net/api/fonts/supersize-bk,sans-serif/font-family:supersize-bk,sans-serif;font-style:normal;font-weight:400;");
      
      :root{
        --bg:#f6f7f9;
        --text:#000000; /* 기본 글씨색 */
        --text-emphasis:#cf65b2; /* 강조 글씨색 */
        --muted:#6b7280;
        --line:rgba(15,23,42,0.10);
        --shadow:0 6px 18px rgba(15, 23, 42, 0.08);
        --radius:18px;
        --btn:#ea580c; /* 포인트 색상 */
        --primary:#febe05; /* 로고 배경색 */
        --logo-text:#000000; /* 로고 텍스트 */
        --logo-font: 'supersize-bk', sans-serif; /* 로고 폰트 */
        
        /* 배경색 */
        --bg-base:#ffffff; /* 기본 배경색 */
        --bg-emphasis:#f1dddd; /* 강조 배경색 */
        
        /* 등급별 색상 */
        --high-title:#dc2626; /* 고위험군 제목 */
        --high-bg:#fee2e2; /* 고위험군 배경 */
        --caution-title:#f5920b; /* 주의 제목 */
        --caution-bg:#fef3c7; /* 주의 배경 */
        --safe-title:#00ad00; /* 안전 제목 */
        --safe-bg:#d1fae5; /* 안전 배경 */
      }

      .stApp { background: var(--bg); }

      .main .block-container{
        max-width: 560px;
        padding-top: 22px;
        padding-bottom: 30px;
      }

      /* hr 제거 */
      hr { display:none !important; }

      .title{
        font-size: 34px;
        font-weight: 900;
        line-height: 1.2;
        color: var(--text);
        margin: 0 0 14px 0;
        white-space: nowrap;
      }

      /* border=True 컨테이너를 카드처럼 */
      div[data-testid="stVerticalBlockBorderWrapper"]{
        background: #fff !important;
        border: 1px solid rgba(15,23,42,0.06) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
        padding: 14px 14px !important;
      }

      .section-label{
        font-size: 22px;
        font-weight: 900;
        color: var(--text);
        display:flex;
        align-items:center;
        gap: 10px;
        margin-bottom: 6px;
      }

      .sub{
        font-size: 14px;
        color: var(--muted);
      }
      
      .calc-card{
        background:#f1dddd !important;
        border-radius:12px;
        padding:18px 16px;
      }


      /* 주소 검색바(secondary 버튼) */
      .addrbar button[kind="secondary"]{
        width: 100% !important;
        height: 62px !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--line) !important;
        background: #fff !important;
        font-size: 17px !important;
        font-weight: 850 !important;
        color: var(--text) !important;
        justify-content: space-between !important;
        transition: background 0.2s ease !important;
      }
      
      /* 주소 검색 버튼 hover */
      .addrbar button[kind="secondary"]:hover{
        background: #f3f4f6 !important;
      }

      .deposit-value{
        font-size: 30px;
        font-weight: 950;
        color: var(--text);
        white-space: nowrap;
      }
      .deposit-won{
        font-size: 14px;
        font-weight: 800;
        color: var(--muted);
        margin-left: 8px;
        white-space: nowrap;
      }

      /* CTA 버튼: primary - 높이 줄이고 텍스트 크게 */
      button[kind="primary"]{
        width: 100% !important;
        min-width: 100% !important;
        height: 72px !important;
        border-radius: 26px !important;
        background: #cf65b2 !important;
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 1000 !important;
        letter-spacing: -0.5px;
        border: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        transition: background 0.2s ease !important;
      }
      
      /* CTA 버튼 hover */
      button[kind="primary"]:hover{
        background: #f1dddd !important;
      }

      div.stButton, div[data-testid="stButton"]{ 
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      div.stButton > button, div[data-testid="stButton"] > button{ 
        width: 100% !important;
        margin: 0 !important;
      }

      /* SLIDER: 트랙 두께 4배로 (34px → 136px), thumb 크기 더 축소 (44px → 32px) */
      div[data-baseweb="slider"]{ padding-top: 14px !important; padding-bottom: 10px !important; }
      div[data-baseweb="slider"] div[role="presentation"]{ height: 136px !important; }
      div[data-baseweb="slider"] div[role="presentation"] > div{ height: 136px !important; border-radius: 999px !important; }
      div[data-testid="stSlider"] div[role="presentation"]{ height: 136px !important; }
      div[data-testid="stSlider"] div[role="presentation"] > div{ height: 136px !important; border-radius: 999px !important; }

      /* thumb - 크기 더 축소 */
      div[data-baseweb="slider"] div[role="slider"]{
        width: 32px !important;
        height: 32px !important;
        border-radius: 999px !important;
        box-shadow: 0 14px 22px rgba(15,23,42,0.28) !important;
      }
      
      /* 슬라이더 하단 입력창 숨기기 */
      div[data-testid="stSlider"] input[type="number"]{
        display: none !important;
      }
      div[data-testid="stSlider"] > div > div:last-child{
        display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Views
# ----------------------------
def render_input():
    # 로고 배너
    st.markdown(
        """
        <div style="background:#febe05; padding:30px 20px; border-radius:12px; margin-bottom:30px; text-align:center;">
            <div style="font-family:'supersize-bk', sans-serif; font-size:56px; font-weight:400; color:#000000; letter-spacing:-2px;">
                내 인생에 전세 사기는 없다
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="title" style="font-size:42px; margin-bottom:30px;">내가 선택한 이 집! 과연 안전할까?</div>', unsafe_allow_html=True)

    # 1) 주소
    with st.container(border=True):
        st.markdown('<div class="section-label">📍 주소</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">주소를 검색하고 선택하세요.</div>', unsafe_allow_html=True)

        display_addr = ""
        if st.session_state.selected_juso:
            display_addr = st.session_state.selected_juso.get("roadAddr", "") or ""
        if not display_addr:
            display_addr = "주소를 검색하세요"

        st.markdown('<div class="addrbar">', unsafe_allow_html=True)
        if st.button(display_addr, type="secondary", key="open_addr"):
            toggle_addr()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.addr_open:
            st.write("")  # spacing
            st.session_state.addr_query = st.text_input(
                "주소 검색",
                value=st.session_state.addr_query,
                placeholder="예) 화곡로 123, 화곡동 1067, OO아파트",
                key="addr_query_input"
            )

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("검색", use_container_width=True, key="addr_search_btn"):
                    st.session_state.addr_page = 1
            with colB:
                if st.button("닫기", use_container_width=True, key="addr_close_btn"):
                    st.session_state.addr_open = False
                    st.rerun()

            q = (st.session_state.addr_query or "").strip()
            if q:
                try:
                    resp = juso_search(q, page=st.session_state.addr_page, count=JUSO_RESULT_PER_PAGE)
                    if not resp["ok"]:
                        st.error(f"주소 검색 오류: {resp['errorMessage']} (code={resp['errorCode']})")
                    else:
                        juso_list = resp["juso"]
                        if not juso_list:
                            st.info("검색 결과가 없어요.")
                        else:
                            st.caption("검색 결과를 선택하세요.")
                            for i, j in enumerate(juso_list):
                                label = j.get("roadAddr", "") or "(주소)"
                                # 보조 정보: 지번
                                jibun = j.get("jibunAddr", "")
                                if jibun:
                                    label = f"{label}  ({jibun})"

                                if st.button(label, type="secondary", key=f"juso_pick_{st.session_state.addr_page}_{i}"):
                                    choose_juso(j)
                                    st.rerun()

                        # pagination
                        total = resp["totalCount"]
                        per = resp["countPerPage"]
                        max_page = max(1, (total + per - 1) // per)

                        pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
                        with pcol1:
                            if st.button("이전", use_container_width=True, disabled=(st.session_state.addr_page <= 1), key="addr_prev"):
                                st.session_state.addr_page -= 1
                                st.rerun()
                        with pcol2:
                            st.caption(f"{st.session_state.addr_page} / {max_page} 페이지  (총 {total}건)")
                        with pcol3:
                            if st.button("다음", use_container_width=True, disabled=(st.session_state.addr_page >= max_page), key="addr_next"):
                                st.session_state.addr_page += 1
                                st.rerun()

                except Exception as e:
                    st.error(f"주소 검색 요청 실패: {e}")
            else:
                st.info("검색어를 입력해 주세요.")

    # 2) 층 / 면적
    with st.container(border=True):
        st.markdown('<div class="section-label">🏢 층 / 면적</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**층**")
            FLOOR_NUM = st.number_input("층수", min_value=-1, max_value=99, value=1, step=1, label_visibility="collapsed", key="floor_input")
            # FLOOR_NUM을 FLOOR 텍스트로 변환
            if FLOOR_NUM == -1:
                FLOOR = "반지하"
            elif FLOOR_NUM == 99:
                FLOOR = "옥탑"
            elif FLOOR_NUM == 1:
                FLOOR = "지상 1층"
            else:
                FLOOR = f"지상 {FLOOR_NUM}층"
        with c2:
            st.markdown("**면적(㎡)**")
            AREA_M2 = st.number_input("면적", min_value=0.0, step=0.5, value=0.0, label_visibility="collapsed", key="area_input")

    # 3) 보증금
    with st.container(border=True):
        top_left, top_right = st.columns([3, 2])
        with top_left:
            st.markdown('<div class="section-label">💰 보증금</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">범위를 조절해 보증금을 설정하세요.</div>', unsafe_allow_html=True)

        DEPOSIT = st.slider(
            "",
            min_value=500,
            max_value=50000,
            value=18000,
            step=100,
            label_visibility="collapsed",
            key="deposit_slider"
        )

        with top_right:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; align-items:baseline; margin-top:32px;">
                  <div class="deposit-value">{DEPOSIT:,}만원</div>
                  <div class="deposit-won">({DEPOSIT*10000:,}원)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4) 계약기간
    with st.container(border=True):
        st.markdown('<div class="section-label">🗓️ 계약기간</div>', unsafe_allow_html=True)
        contract_label = st.selectbox(
            "",
            ["1년", "2년", "3년", "4년 이상"],
            index=1,
            label_visibility="collapsed",
            key="contract_select"
        )
        contract_years = parse_contract_years(contract_label)

    # 5) CTA - 좌우 꽉 차게
    st.markdown('<div style="margin: 0 -14px;">', unsafe_allow_html=True)
    clicked = st.button("이 조건으로 위험도 확인하기", type="primary", key="cta_to_result", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        import numpy as np
        import pandas as pd
        import tracka_final as ta
        import trackb_final as tb
        
        selected = st.session_state.selected_juso or {}

        # --- 지번번호(JIBUN) 가공 ---
        main_no = str(selected.get("lnbrMnnm", "")).strip()
        sub_no  = str(selected.get("lnbrSlno", "0")).strip()

        if sub_no in ["", "0", "0000"]:
            JIBUN = main_no
        else:
            JIBUN = f"{main_no}-{int(float(sub_no))}"

        if not JIBUN:
            import re
            m = re.search(r"(\d+)(?:-(\d+))?\s*$", str(selected.get("jibunAddr", "")).strip())
            if m:
                JIBUN = m.group(1) if not m.group(2) else f"{m.group(1)}-{m.group(2)}"
            else:
                JIBUN = ""

        st.session_state.inputs = {
            "JIBUN": JIBUN,
            "AREA_M2": float(AREA_M2),
            "FLOOR" : FLOOR,
            "FLOOR_NUM": int(FLOOR_NUM) if FLOOR_NUM != 99 and FLOOR_NUM != -1 else (0 if FLOOR_NUM == -1 else 99),
            "DEPOSIT": int(DEPOSIT),
            "ROAD_ADDR": selected.get("roadAddr", ""),
            "ZIPNO": selected.get("zipNo", ""),
            "CONTRACT_YEARS": int(contract_years),
        }

        # ✅ Track A 계산
        try:
            with st.spinner("Track A 계산 중..."):
                resA, commentsA = ta.predict_final(
                    jibun=JIBUN,
                    area_m2=float(AREA_M2),
                    floor=int(floor_to_num(FLOOR)),
                    deposit=int(DEPOSIT),
                )
                st.session_state.inputs["TRACKA_RESULT"] = resA
                st.session_state.inputs["TRACKA_COMMENTS"] = commentsA
                st.session_state.inputs["V0"] = float(resA.get("V0", np.nan))
        except Exception as e:
            st.error(f"Track A 계산 실패: {e}")
            st.stop()

        # ✅ Track B 계산
        V0 = st.session_state.inputs.get("V0")
        if V0 and not np.isnan(V0):
            try:
                with st.spinner("Track B 계산 중..."):
                    B = float(DEPOSIT)
                    T = float(contract_years)
                    
                    df_in = pd.DataFrame([{
                        "hedonic_price": float(V0),
                        "deposit": B,
                        "term": T,
                    }])
                    
                    df_out = tb.add_trackB_risk_columns(
                        df_in,
                        v0_col="hedonic_price",
                        b_col="deposit",
                        t_col="term",
                        mu=tb.MU_ANNUAL,
                        sigma=tb.SIGMA_ANNUAL,
                        alpha=tb.ALPHA_USED,
                        scenarios=tb.SCENARIOS
                    )
                    
                    row = df_out.iloc[0]
                    st.session_state.inputs["TRACKB_RESULT"] = row.to_dict()
                    st.session_state.inputs["JEONSE_RATIO"] = float(row["jeonse_ratio"])
            except Exception as e:
                st.error(f"Track B 계산 실패: {e}")
                st.stop()

        go("result")



def render_result():
    st.markdown('<div class="title" style="font-size:42px; margin-bottom:30px;">내가 고른 집의 점수는...</div>', unsafe_allow_html=True)

    inputs = st.session_state.get("inputs", {})

    # ---- Track A/Track B 결과 꺼내기 ----
    resA = inputs.get("TRACKA_RESULT", {}) or {}
    resB = inputs.get("TRACKB_RESULT", {}) or {}

    probA = resA.get("prob", None)
    # TrackB는 너가 지금 화면에서 PD_base를 쓰고 있으니 그걸 사분면 점수로 사용
    probB = resB.get("PD_base", None)

    # ---- 아직 계산 안 된 경우 안내 ----
    if probA is None:
        st.warning("Track A 결과가 아직 없어요. Track A 페이지에서 먼저 계산해 주세요.")
        if st.button("Track A로 가기", key="go_trackA_from_result"):
            go("trackA")
        return

    if probB is None:
        st.warning("Track B 결과가 아직 없어요. Track B 페이지에서 먼저 계산해 주세요.")
        if st.button("Track B로 가기", key="go_trackB_from_result"):
            go("trackB")
        return

    # ---- 등급(3단계 분류) ----
    a_grade = classify_3bin(float(probA), TRACKA_T1, TRACKA_T2)
    b_grade = classify_3bin(float(probB), TRACKB_T1, TRACKB_T2)

    # 9분면 케이스 매핑
    zone_code, zone_name, zone_desc, zone_bg, zone_color = get_9zone_case(a_grade, b_grade)

    # ---- UI 출력 ----
    # 전체를 하나의 container로 감싸기
    with st.container(border=True):
        st.markdown('<div style="font-size:28px; font-weight:900; margin-bottom:8px;">종합 위험도 평가 (9분면)</div>', unsafe_allow_html=True)
        st.caption("Track A와 Track B를 교차 분석한 결과입니다.")

    st.markdown(
        """
        <style>
          .zone9 { 
            width:100%; 
            border-collapse:collapse; 
            margin-top:16px; 
            font-size:20px;
          }
          .zone9 th, .zone9 td { 
            border:1px solid rgba(15,23,42,0.12); 
            padding:16px 12px; 
            text-align:center; 
            font-weight:700;
            color:#000000;
          }
          .zone9 th { 
            background: rgba(15,23,42,0.04); 
            font-weight:800;
            font-size:18px;
          }
          .zone9 .selected { 
            font-weight:900;
          }
          .zone9 .selected-green {
            outline: 3px solid #00ad00; 
            background: #d1fae5;
          }
          .zone9 .selected-orange {
            outline: 3px solid #f5920b; 
            background: #fef3c7;
          }
          .zone9 .selected-red {
            outline: 3px solid #dc2626; 
            background: #fee2e2;
          }
          .zone9 .zone-num { 
            font-size:24px; 
            font-weight:900; 
            display:block;
            margin-bottom:4px;
          }
          .zone9 .zone-name { 
            font-size:16px; 
            font-weight:700; 
            color:#000000;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 현재 선택된 셀 판별 및 색상 결정
    def get_cell_class(a_val, b_val):
        if a_val != a_grade or b_val != b_grade:
            return ""
        
        # zone_code에 따라 색상 결정
        if zone_code == "①":
            return "selected selected-green"
        elif zone_code == "⑨":
            return "selected selected-red"
        else:  # ②~⑧
            return "selected selected-orange"

    st.markdown(
        f"""
        <table class="zone9">
          <tr>
            <th>Track A \\ Track B</th>
            <th>Safe<br>(저위험)</th>
            <th>Caution<br>(주의)</th>
            <th>High<br>(고위험)</th>
          </tr>
          <tr>
            <th>Safe<br>(안전)</th>
            <td class="{get_cell_class('Safe','Safe')}">
              <span class="zone-num">①</span>
              <span class="zone-name">최적 안전존</span>
            </td>
            <td class="{get_cell_class('Safe','Caution')}">
              <span class="zone-num">②</span>
              <span class="zone-name">시장 관찰존</span>
            </td>
            <td class="{get_cell_class('Safe','High')}">
              <span class="zone-num">③</span>
              <span class="zone-name">시장 경고존</span>
            </td>
          </tr>
          <tr>
            <th>Caution<br>(주의)</th>
            <td class="{get_cell_class('Caution','Safe')}">
              <span class="zone-num">④</span>
              <span class="zone-name">패턴 주의존</span>
            </td>
            <td class="{get_cell_class('Caution','Caution')}">
              <span class="zone-num">⑤</span>
              <span class="zone-name">복합 관리존</span>
            </td>
            <td class="{get_cell_class('Caution','High')}">
              <span class="zone-num">⑥</span>
              <span class="zone-name">리스크 심화존</span>
            </td>
          </tr>
          <tr>
            <th>High<br>(고위험)</th>
            <td class="{get_cell_class('High','Safe')}">
              <span class="zone-num">⑦</span>
              <span class="zone-name">사기 경고존</span>
            </td>
            <td class="{get_cell_class('High','Caution')}">
              <span class="zone-num">⑧</span>
              <span class="zone-name">위험 확산존</span>
            </td>
            <td class="{get_cell_class('High','High')}">
              <span class="zone-num">⑨</span>
              <span class="zone-name">절대 금지존</span>
            </td>
          </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

        # 9분면 결과 카드 (container 없이, margin-top 추가)
    st.markdown(
            f"""
            <div style="background:{zone_bg}; padding:24px; border-radius:12px; border-left:6px solid {zone_color}; margin-top:20px;">
                <div style="font-size:30px; font-weight:900; color:{zone_color}; margin-bottom:12px;">
                    {zone_code} {zone_name}
                </div>
                <div style="font-size:16px; font-weight:600; color:#000000; line-height:1.6;">
                    {zone_desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # 종합 위험도 평가 container 닫기

    st.markdown("---")
    
    # ---- 전세가율 (9분면 뒤에 표시) ----
    
    st.divider()
    with st.container(border=True):
        st.markdown('<div style="font-size:28px; font-weight:900; margin-bottom:12px;">전세가율</div>', unsafe_allow_html=True)

        V0 = inputs.get("V0", None)
        deposit = inputs.get("DEPOSIT", None)

        ratio = jr.calc_jeonse_ratio(deposit, V0)

        if ratio is None:
            st.warning("전세가율을 계산할 수 없어요. (적정 매매가 없음)")
        else:
            st.metric(
                label="전세가율 (보증금 / 적정 매매가)",
                value=f"{ratio:.1%}",
                help="전세금이 적정 매매가에서 차지하는 비율이에요"
            )
            # 세션에 저장
            inputs["JEONSE_RATIO"] = float(ratio)
            st.session_state.inputs = inputs

    st.markdown("---")

    # 상세 페이지 이동 버튼 2개 - 좌우 꽉 차게
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        if st.button("Track A 값 확인하러 가기", key="goA_from_result", use_container_width=True):
            go("trackA")
    with c2:
        if st.button("Track B 값 확인하러 가기", key="goB_from_result", use_container_width=True):
            go("trackB")

    st.write("")
    if st.button("⬅︎ 입력 화면으로 돌아가기", use_container_width=True, key="back_to_input"):
        go("input")


def render_trackA():
    import numpy as np
    import tracka_final as ta
    import plotly.graph_objects as plotly_go

    def _is_nan(v):
        try:
            return v is None or (isinstance(v, float) and np.isnan(v))
        except Exception:
            return v is None

    st.markdown('<div style="font-size:42px; font-weight:900; margin-bottom:8px;">Track A: 전세사기 위험 분석</div>', unsafe_allow_html=True)
    st.caption("과거 사기 패턴과 유사한지 알려드릴게요.")

    inputs = st.session_state.get("inputs", {})

    # 입력값 표시
    with st.container(border=True):
        st.markdown("### 현재 입력값")
        
        # 테이블 형식으로 깔끔하게 정리
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:12px; margin-bottom:24px;">
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">주소</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('ROAD_ADDR', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">보증금</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('DEPOSIT', 0):,}만원</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">층</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('FLOOR', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">계약기간</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('CONTRACT_YEARS', 0)}년</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">면적</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('AREA_M2', 0):.2f}㎡</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">지번</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('JIBUN', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 이미 계산된 값 재사용
    if "TRACKA_RESULT" in inputs and "TRACKA_COMMENTS" in inputs and (not _is_nan(inputs.get("V0", None))):
        resA = inputs["TRACKA_RESULT"]
        commentsA = inputs["TRACKA_COMMENTS"]
    else:
        try:
            with st.spinner("Track A 계산 중..."):
                resA, commentsA = ta.predict_final(
                    jibun=inputs["JIBUN"],
                    area_m2=float(inputs["AREA_M2"]),
                    floor=int(inputs.get("FLOOR_NUM", 1)),
                    deposit=int(inputs["DEPOSIT"]),
                )

            inputs["TRACKA_RESULT"] = resA
            inputs["TRACKA_COMMENTS"] = commentsA
            inputs["V0"] = float(resA.get("V0", np.nan))
            st.session_state.inputs = inputs

        except Exception as e:
            st.error("Track A 계산 중 오류가 발생했어요.")
            st.exception(e)
            return

    # ============================================
    # Main Top: 종합 리스크 등급 및 게이지 차트
    # ============================================
    prob_value = float(resA.get("prob", 0))
    v0_value = resA.get("V0", 0)
    
    # 등급 판정 (임계값 통일: 56%, 68%)
    if prob_value < 0.56:
        grade_display = "Safe"
        grade_korean = "안전"
        grade_color = "#00ad00"
        grade_bg = "#d1fae5"
        gauge_color = "#00ad00"
    elif prob_value < 0.68:
        grade_display = "Caution"
        grade_korean = "주의"
        grade_color = "#f5920b"
        grade_bg = "#fef3c7"
        gauge_color = "#f5920b"
    else:
        grade_display = "High"
        grade_korean = "고위험"
        grade_color = "#dc2626"
        grade_bg = "#fee2e2"
        gauge_color = "#dc2626"

    with st.container(border=True):
        st.markdown("### 🎯 종합 리스크 등급 및 사고 확률")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 게이지 차트 (전면 수정)
            fig = plotly_go.Figure(plotly_go.Indicator(
                mode="gauge+number",
                value=prob_value * 100,
                title={'text': "경매 발생 예측 확률", 'font': {'size': 18, 'color': '#000000'}},
                number={'suffix': "%", 'font': {'size': 48, 'color': '#000000'}},
                gauge={
                    'axis': {
                        'range': [0, 100], 
                        'tickwidth': 1, 
                        'tickcolor': '#000000',
                        'tickfont': {'color': '#000000'}
                    },
                    'bar': {'color': gauge_color, 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#d1fae5'},
                        {'range': [50, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': gauge_color, 'width': 4},
                        'thickness': 0.75,
                        'value': prob_value * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'family': "Arial", 'color': '#000000'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 등급 카드 (폰트 크기/색상 완전히 교체)
            current_deposit = float(inputs.get('DEPOSIT', 0))
            st.markdown(
                f"""
                <div style="background:{grade_bg}; padding:24px; border-radius:12px; height:280px; display:flex; flex-direction:column; justify-content:center; border-left:6px solid {grade_color};">
                    <div style="font-size:18px; font-weight:700; color:#000000; margin-bottom:12px;">위험 등급</div>
                    <div style="font-size:48px; font-weight:900; color:{grade_color}; margin-bottom:16px;">
                        {grade_korean}
                    </div>
                    <div style="font-size:18px; font-weight:700; color:#000000; margin-top:8px;">
                        적정 매매가: {v0_value:,.0f}만원
                    </div>
                    <div style="font-size:18px; font-weight:700; color:#000000; margin-top:4px;">
                        현재 보증금: {current_deposit:,.0f}만원
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ============================================
    # Main Middle: 3대 핵심 지표 (Left) + XAI 차트 (Right)
    # ============================================
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # 3대 핵심 지표 수치 카드
        with st.container(border=True):
            st.markdown("### 📊 3대 핵심 사기 패턴")
            
            # resA에서 logistic_features 가져오기
            logistic_features_dict = resA.get("logistic_features", {})
            
            # 기본값 설정
            deposit = float(inputs.get('DEPOSIT', 0))
            
            if v0_value and v0_value > 0:
                deposit_overhang = deposit - v0_value
            else:
                deposit_overhang = 0
            
            # logistic_features에서 실제 값 가져오기
            if logistic_features_dict:
                deposit_overhang = logistic_features_dict.get("deposit_overhang", deposit_overhang)
                nearby_auction = int(logistic_features_dict.get("nearby_auction_1km", 0))
                local_morans_i = logistic_features_dict.get("local_morans_i", 0)
            else:
                nearby_auction = 0
                local_morans_i = 0
            
            st.markdown(
                f"""
                <div style="padding:12px; border-radius:8px; margin-bottom:16px;">
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">적정 매매가에 비해 이만큼 더 비싸요</div>
                    <div style="font-size:28px; font-weight:900; color:#000000;">{deposit_overhang:,.0f} 만원</div>
                </div>
                
                <div style="padding:12px; border-radius:8px; margin-bottom:16px;">
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">주변 매물과의 가격 차이가 이만큼 있어요</div>
                    <div style="font-size:28px; font-weight:900; color:#000000;">{local_morans_i:.4f}</div>
                </div>
                
                <div style="padding:12px; border-radius:8px; margin-bottom:16px;">
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">주변에서 경매가 이만큼 발생했어요</div>
                    <div style="font-size:28px; font-weight:900; color:#000000;">{nearby_auction}건</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col_right:
        # XAI 차트 (WOE 기반)
        with st.container(border=True):
            st.markdown("### 🔍 사기 패턴 유사 항목 분석")
            st.caption("어떤 것이 사기 패턴과 비슷한지 알려드릴게요")
            
            st.markdown(
                """
                <div style="padding:16px; background:#f1dddd; border-left:4px solid #cf65b2; border-radius:8px; margin-bottom:12px;">
                    <div style="font-size:14px; font-weight:700; color:#000000; margin-bottom:8px;">
                        ⚠️ WOE 값 기반 사기 패턴 유사도
                    </div>
                    <div style="font-size:13px; color:#000000; line-height:1.6;">
                        WOE 값을 이용해서 어떤 요소가 속한 구간의 패턴이 과거 사기 사례와 유사한지 보여드릴게요
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 실제 WOE 값 사용
            woe_values_dict = resA.get("woe_values", {})
            
            if woe_values_dict:
                # WOE 값을 정규화 (상대적 비중)
                feature_names_mapping = {
                    "deposit_overhang": "보증금 초과액",
                    "effective_LTV": "전세가율",
                    "local_morans_i": "공간적 이상치",
                    "nearby_auction_1km": "주변 경매"
                }
                
                feature_names = []
                woe_values = []
                
                for key, value in woe_values_dict.items():
                    feature_names.append(feature_names_mapping.get(key, key))
                    woe_values.append(abs(value))  # 절대값 사용
                
                # 정규화 (합이 1이 되도록)
                total = sum(woe_values) if sum(woe_values) > 0 else 1
                woe_values_normalized = [v / total for v in woe_values]
                
                fig_xai = plotly_go.Figure(data=[
                    plotly_go.Bar(
                        x=woe_values_normalized,
                        y=feature_names,
                        orientation='h',
                        marker=dict(
                            color=['#fee2e2' if v > 0.3 else '#fef3c7' if v > 0.15 else '#d1fae5' for v in woe_values_normalized],
                            line=dict(color='#000000', width=1)
                        ),
                        text=[f"{v:.1%}" for v in woe_values_normalized],
                        textposition='auto',
                        textfont=dict(color='#000000')
                    )
                ])
                
                fig_xai.update_layout(
                    title=dict(text="요소별 사기 패턴 유사도", font=dict(color='#000000')),
                    xaxis=dict(title="유사도", titlefont=dict(color='#000000'), tickfont=dict(color='#000000')),
                    yaxis=dict(tickfont=dict(color='#000000')),
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=40),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig_xai, use_container_width=True)
            else:
                st.info("WOE 값을 계산할 수 없습니다.")
    
    # ============================================
    # 요약 및 행동강령 (종합 리스크 등급 아래로 이동)
    # ============================================
    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("### 💬 요약 및 행동강령")
        
        # 등급별 요약 및 행동강령
        if grade_display == "Safe":  # 안전
            st.markdown(
                """
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[요약]</div>
                
                **안전('신뢰 패턴' 확인)**: 본 매물은 과거 사기 매물과 패턴이 유사하지 않은 '클린 매물'입니다. 예측 매매가 대비 보증금이 안정적이며, 주변 지역의 사고 이력도 낮습니다.
                
                <hr style="margin: 20px 0;">
                
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[행동강령]</div>
                
                **[안전('신뢰 패턴' 확인)] 단계: "통상적 절차 진행 및 모니터링"**
                
                • **표준 권리분석**: 등기부등본상 선순위 채권 유무 등 기본적인 권리관계를 최종 점검하십시오.

                • **시장 변화 주시**: 현재는 안정하나, Track B 시나리오 분석을 통해 향후 급격 인상이나 시장 급변 시에도 보증금이 안전할지 한 번 더 체크하십시오.

                • **보증보험 가입**: 안전 등급이라 하더라도 만약의 사고를 대비해 전세보증보험 가입을 권장합니다.
                """,
                unsafe_allow_html=True
            )
        elif grade_display == "Caution":  # 주의
            st.markdown(
                """
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[요약]</div>
                
                **주의('주의 패턴' 포착)**: 본 매물은 과거 사기 패턴과 비슷한 점들이 포착되었습니다.
                
                <hr style="margin: 20px 0;">
                
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[행동강령]</div>
                
                **[주의('주의 패턴' 포착)] 단계: "정밀 확인 및 계약 조건 협상"**
                
                • **보증금 하향 협상**: Track B에서 제시하는 '적정 보증금' 수치를 확인하고, 해당 금액 이하로 보증금을 조정할 것을 권장합니다.

                • **특약 사항 추가**: "임대인은 전금 지급일 다음 날까지 담보권을 설정하지 않는다" 등 전세 사기 방지 표준 특약을 반드시 계약서에 명시하십시오.

                • **Track B 결과 확인**: 의도적 사기 외에 항후 집값 하락에 따른 '시장 리스크'가 얼마나 있는지 추가로 확인하십시오.
                """,
                unsafe_allow_html=True
            )
        else:  # High (고위험)
            st.markdown(
                """
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[요약]</div>
                
                **고위험('위험 패턴' 일치)**: 본 매물은 강력한 사기 의도가 의심되는 단계입니다. 경매 사고 확률이 매우 높습니다."
                
                <hr style="margin: 20px 0;">
                
                <div style="font-size:20px; font-weight:900; margin-bottom:12px;">[행동강령]</div>
                
                **[고위험('위험 패턴' 일치)] 단계: "계약 재검토 및 강력 주의"**
                
                • **계약 보류 권고**: 해당 매물은 4년 내 경매 발생 확률이 통계적으로 매우 높으므로, 계약 체결을 재검토할 것을 강력히 권고합니다.

                • **법인 여부 확인**: 집주인이 법인일 경우, 절값 매입 후 보증금을 가로채는 '기획 사기'일 가능성이 높으므로 반드시 법인 등기부와 재무 상태를 확인하십시오.

                • **특약 사항 추가**: "임대인은 잔금 지급일 다음 날까지 담보권을 설정하지 않는다" 등 전세 사기 방지 표준 특약을 반드시 계약서에 명시하십시오.

                • **보증보험 필수**: 계약을 진행할 경우 반드시 HUG 전세보증금반환보증 가입 가능 여부를 확인하고, 불가능할 경우 계약하지 마십시오.
                """,
                unsafe_allow_html=True
            )
    
    # ============================================
    # 계산 과정 토글
    # ============================================
    st.markdown("---")
    
    # 토글 상태 초기화
    if "show_tracka_calc" not in st.session_state:
        st.session_state.show_tracka_calc = False
    
    if st.button("▼ 이 결과는 어떻게 계산됐나요?" if not st.session_state.show_tracka_calc else "▲ 이 결과는 어떻게 계산됐나요?", key="toggle_tracka_calc"):
        st.session_state.show_tracka_calc = not st.session_state.show_tracka_calc
    
    if st.session_state.show_tracka_calc:
        st.markdown(
            """
<div style="background:#f1dddd; padding:24px; border-radius:12px; border:1px solid #e5e7eb; margin:16px 0;">
    <h3 style="color:#000000; margin-top:0;">① 정상 전세가는 어떻게 계산했나요?</h3>
    
    <p style="color:#000000;">이 집이 정상적인 시장 상황이라면 얼마의 전세가가 적정한지 먼저 계산했어요.</p>
    
    <ul style="color:#000000;">
        <li>면적, 층수, 건물 연식, 위치 같은 집의 물리적 특성을 사용했어요.</li>
        <li>같은 동네(용·면·동 단위)에서 실제 거래된 전세 데이터를 바탕으로 계산했어요.</li>
        <li>이를 통해 이 집의 정상 전세가를 추정치를 구했어요.</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">② 실제 전세가와 얼마나 차이가 나나요?</h3>
    
    <p style="color:#000000;">실제 전세가가 정상 전세가와 얼마나 다른지를 봤어요.</p>
    
    <ul style="color:#000000;">
        <li>실제 전세가 − 정상 전세가 = 가격 차액</li>
        <li>이 차이가 크면, 시장 가격과 어긋난 신호일 수 있어요.</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">③ 주변 매물들과 비교했을 때 이상한가요?</h3>
    
    <p style="color:#000000;">혼자만 뛰는 건지, 주변도 다 비슷한지 확인했어요.</p>
    
    <ul style="color:#000000;">
        <li>반경 r km 이내 매물들과 가격 차이를 비교했어요.</li>
        <li>주변도 비슷하면 → 시장 전체 오인!</li>
        <li>이 집만 튀면 → 정보 비대칭 또는 비정상 신호</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">④ 그래서 Track A 점수는 뭔가요?</h3>
    
    <p style="color:#000000;">최종적으로, 이 집의 가격이 주변 대비 얼마나 비정상적인지를 점수로 만들었어요.</p>
    
    <ul style="color:#000000;">
        <li>가격 차이가 클수록 점수가 커져요.</li>
        <li>점수는 확률이 아니라 가격 왜곡 정도예요.</li>
        <li>점수가 높을수록 주의가 필요한 매물이에요.</li>
    </ul>
</div>
            """,
            unsafe_allow_html=True
        )

    if st.button("⬅︎ 요약으로 돌아가기", use_container_width=True):
        go("result")


def render_trackB():
    import numpy as np
    import pandas as pd
    import trackb_final as tb

    st.markdown('<div style="font-size:42px; font-weight:900; margin-bottom:8px;">Track B: 시장 리스크 분석</div>', unsafe_allow_html=True)
    st.caption("계약 만기 시점의 시장 상황에 따라 보증금을 돌려받을 수 있는지 알려줘요.")

    inputs = st.session_state.get("inputs", {})

    # 입력값 표시
    with st.container(border=True):
        st.markdown("### 현재 입력값")
        
        # 테이블 형식으로 깔끔하게 정리 (Track A와 동일)
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:12px; margin-bottom:24px;">
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">주소</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('ROAD_ADDR', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">보증금</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('DEPOSIT', 0):,}만원</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">층</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('FLOOR', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">계약기간</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('CONTRACT_YEARS', 0)}년</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">면적</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('AREA_M2', 0):.2f}㎡</div>
                </div>
                <div>
                    <div style="font-size:14px; color:#6b7280; margin-bottom:4px;">지번</div>
                    <div style="font-size:16px; font-weight:700; color:#000000;">{inputs.get('JIBUN', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Track A 결과(V0) 없으면 안내
    V0 = inputs.get("V0", None)
    if V0 is None or (isinstance(V0, float) and np.isnan(V0)):
        st.warning("Track B를 계산하려면 Track A에서 산출된 적정 매매가가 필요해요. Track A를 먼저 실행해 주세요.")
        if st.button("⬅︎ 요약으로 돌아가기", use_container_width=True):
            go("result")
        return

    # Track B 입력값 구성
    B = float(inputs["DEPOSIT"])
    T = float(inputs["CONTRACT_YEARS"])

    df_in = pd.DataFrame([{
        "hedonic_price": float(V0),
        "deposit": B,
        "term": T,
    }])

    # 계산
    try:
        with st.spinner("Track B 계산 중..."):
            df_out = tb.add_trackB_risk_columns(
                df_in,
                v0_col="hedonic_price",
                b_col="deposit",
                t_col="term",
                mu=tb.MU_ANNUAL,
                sigma=tb.SIGMA_ANNUAL,
                alpha=tb.ALPHA_USED,
                scenarios=tb.SCENARIOS
            )

            rep, base_el, el_20, slope = tb.scenario_sensitivity_report(df_out, idx=0, make_plot=False)

            # B* (적정보증금 상한)
            shock = tb.SCENARIOS.get(tb.SCENARIO_FOR_BSTAR, 0.0)
            b_before, b_after = tb.B_star_range_two_mu(
                V0=float(V0),
                T=float(T),
                sigma=tb.SIGMA_ANNUAL,
                alpha=tb.ALPHA_USED,
                shock=shock,
                EL_CAP=tb.EL_CAP,
                mu_before=tb.MU_HAT,
                mu_after=tb.MU_ANNUAL,
                tol=100.0
            )

    except Exception as e:
        st.error("Track B 계산 중 오류가 발생했어요.")
        st.exception(e)
        return

    row = df_out.iloc[0]
    inputs["TRACKB_RESULT"] = row.to_dict()
    inputs["JEONSE_RATIO"] = float(row["jeonse_ratio"])
    st.session_state.inputs = inputs

    # === 금융 리스크 등급 계산 ===
    # 기준금리 1%p 상승 시 예상 손실액 계산
    el_base = float(row['EL_base'])
    el_20 = float(row['EL_stress20'])
    el_change_per_1pct = (el_20 - el_base) / 20.0
    
    # 현재 보증금 대비 손실 비율
    current_deposit = float(inputs["DEPOSIT"])
    loss_ratio = (el_change_per_1pct / current_deposit * 100) if current_deposit > 0 else 0
    
    # 등급 판정
    if loss_ratio < 5:
        financial_grade = "A"
        grade_color_fin = "#10b981"  # 녹색
    elif loss_ratio < 10:
        financial_grade = "B"
        grade_color_fin = "#f59e0b"  # 주황색
    else:
        financial_grade = "C"
        grade_color_fin = "#ef4444"  # 빨간색

    # === 등급 계산 ===
    pd_value = row['PD_base']
    
    b_grade = classify_3bin(float(pd_value), TRACKB_T1, TRACKB_T2)
    is_deposit_over = current_deposit > b_after
    
    # 등급별 메시지
    if b_grade == "High":
        risk_level = "⚠️ 고위험"
        risk_bg = "#fee2e2"
        risk_color = "#dc2626"
        risk_message = "본 계약은 금융적 부도 위험이 매우 높은 상태입니다. 현재 보증금이 집값의 변동성을 충분히 방어하지 못하고 있으며, 시장 하락 시 대규모 자산 손실이 예상됩니다."
        risk_items = [
            "손실 확률이 높은 수준이에요",
            "집값 변동성을 주의깊게 모니터링하세요",
            "경매 시 낙찰가가 감정가보다 낮게 형성될 수 있어요"
        ]
    elif b_grade == "Caution":
        risk_level = "⚠️ 주의"
        risk_bg = "#fef3c7"
        risk_color = "#f5920b"
        risk_message = "본 계약은 현재 환경에서는 안정적이나, 시장 충격에 다소 취약합니다. 집값이 10% 이상 하락하거나 금리가 급등할 경우 보증금 일부를 돌려받지 못할 기대 손실(EL)이 발생할 가능성이 있습니다."
        risk_items = [
            "손실 확률이 다소 높은 수준이에요",
            "집값 변동성을 주의깊게 모니터링하세요",
            "경매 시 낙찰가가 감정가보다 낮게 형성될 수 있어요"
        ]
    else:  # Safe
        risk_level = "✅ 안전"
        risk_bg = "#d1fae5"
        risk_color = "#00ad00"
        risk_message = "본 계약은 현재 시장 상황에서 매우 높은 금융적 안정성을 보이고 있습니다. 역사적 통계에 따르면 이 등급의 매물은 실제 보증금 미반환 사고가 발생하지 않은 '제로 리스크' 구역에 해당합니다."
        risk_items = [
            "손실 확률이 낮은 수준이에요",
            "보증금이 적정 범위 이내에요",
            "비교적 안전한 조건입니다"
        ]

    # ============================================
    # 1. 요약 및 행동강령 (맨 위 노란색 카드)
    # ============================================
    with st.container(border=True):
        st.markdown('<div style="font-size:28px; font-weight:900; margin-bottom:12px;">이 전세, 돈을 잃을 확률은 얼마인가요?</div>', unsafe_allow_html=True)
        
        risk_items_html = "".join([f"<li>{item}</li>" for item in risk_items])
        st.markdown(
            f"""
            <div style="background:{risk_bg}; padding:16px; border-radius:12px; margin:12px 0;">
                <div style="font-size:18px; font-weight:800; color:{risk_color}; margin-bottom:8px;">{risk_level}</div>
                <div style="color:#000000; font-weight:600; line-height:1.6; margin-bottom:12px;">{risk_message}</div>
                <div style="margin-top:12px; color:#000000; font-weight:600;">주요 요소 요약</div>
                <ul style="margin:8px 0; padding-left:20px; color:#000000;">
                    {risk_items_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================
    # 2. 종합 리스크 요약
    # ============================================
    with st.container(border=True):
        st.markdown('<div style="font-size:28px; font-weight:900; margin-bottom:12px;">📊 종합 리스크 요약</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div style="text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#163a66; margin-bottom:16px;">보증금을 못 돌려받을 확률</div>
                    <div style="font-size:36px; font-weight:900; color:#000000;">{PD_VALUE}</div>
                </div>
                """.replace("{PD_VALUE}", f"{row['PD_base']:.1%}"),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div style="text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#163a66; margin-bottom:16px;">평균적으로 잃을 수 있는 금액</div>
                    <div style="font-size:36px; font-weight:900; color:#000000;">약 {EL_VALUE}만원</div>
                </div>
                """.replace("{EL_VALUE}", f"{row['EL_base']:,.0f}"),
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div style="text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#163a66; margin-bottom:16px;">금융 리스크 등급</div>
                    <div style="font-size:36px; font-weight:900; color:#000000;">{GRADE}</div>
                </div>
                """.replace("{GRADE}", financial_grade),
                unsafe_allow_html=True
            )

    # ============================================
    # 3. 시장 시나리오별 리스크 분석
    # ============================================
    # 추가 변수 (el_10)
    el_10 = float(row.get('EL_stress10', 0))
    
    with st.container(border=True):
        st.markdown('<div style="font-size:28px; font-weight:900; margin-bottom:12px;">📈 시장 시나리오별 리스크 분석</div>', unsafe_allow_html=True)
        
        # 금리 영향도
        st.markdown(f"**💡 금리 영향도**: 기준금리 1%p 상승할 때 예상 손실액이 약 {el_change_per_1pct:,.0f}만원씩 증가합니다.")
        st.markdown(f"**📊 가격 변동성**: 화곡동의 연간 가격 변동성은 {tb.SIGMA_ANNUAL*100:.2f}%예요.")
        
        st.markdown("---")
        
        # 표 생성
        st.markdown("#### 시나리오별 위험 지표")
        scenario_data = {
            "시장 시나리오": ["정상 (0%)", "-10% 하락", "-20% 하락"],
            "PD (손실 확률)": [
                f"{row['PD_base']:.2%}",
                f"{row.get('PD_stress10', 0):.2%}",
                f"{row['PD_stress20']:.2%}"
            ],
            "LGD (손실률)": [
                f"{row['LGD_base']:,.0f}만원",
                f"{row.get('LGD_stress10', 0):,.0f}만원",
                f"{row['LGD_stress20']:,.0f}만원"
            ],
            "EL (예상 평균 손실)": [
                f"{el_base:,.0f}만원",
                f"{el_10:,.0f}만원",
                f"{el_20:,.0f}만원"
            ]
        }
        
        df_scenario = pd.DataFrame(scenario_data)
        st.dataframe(df_scenario, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 꺾은선 그래프 (Plotly 사용)
        st.markdown("#### 시나리오별 예상 손실 변화")
        
        import plotly.graph_objects as plotly_go
        
        fig_scenario = plotly_go.Figure()
        
        fig_scenario.add_trace(plotly_go.Scatter(
            x=["정상 (0%)", "-10% 하락", "-20% 하락"],
            y=[el_base, el_10, el_20],
            mode='lines+markers',
            name='예상 평균 손실',
            line=dict(color='#163a66', width=3),
            marker=dict(size=10, color='#163a66')
        ))
        
        fig_scenario.update_layout(
            xaxis=dict(
                title="시장 시나리오",
                titlefont=dict(color='#000000'),
                tickfont=dict(color='#000000')
            ),
            yaxis=dict(
                title="예상 평균 손실 (만원)",
                titlefont=dict(color='#000000'),
                tickfont=dict(color='#000000')
            ),
            height=400,
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=50, r=50, t=30, b=50)
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)

    # ============================================
    # 4. 적정 보증금 범위
    # ============================================
    if current_deposit <= b_after:
        safety_status = "안전"
        safety_color = "#00ad00"
        safety_message = f"현재 보증금({current_deposit:,.0f}만원)은 적정 범위 이하로 안전합니다."
    else:
        safety_status = "위험"
        safety_color = "#dc2626"
        over_amount = current_deposit - b_after
        safety_message = f"현재 보증금({current_deposit:,.0f}만원)이 적정 범위를 {over_amount:,.0f}만원 초과합니다. 위험할 수 있습니다."
    
    with st.container(border=True):
        st.markdown("### ❓ 적정 보증금 범위 분석")
        st.markdown(
            f"""
            <div style="border:2px solid #e5e7eb; padding:16px; border-radius:8px; background:#f9fafb; margin-bottom:20px;">
                <div style="font-weight:800; margin-bottom:8px;">[적정 보증금 범위]</div>
                <div style="font-size:20px; font-weight:900; color:#163a66; margin-bottom:12px;">
                    {b_before:,.0f} 만원 ~ {b_after:,.0f} 만원
                </div>
                <div style="margin-top:8px; padding:12px; border-radius:8px; background:{safety_color}15; border-left:4px solid {safety_color};">
                    <div style="font-weight:800; color:{safety_color}; margin-bottom:4px;">
                        {safety_status} 상태
                    </div>
                    <div style="color:#374151; font-weight:600;">
                        {safety_message}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ============================================
    # 계산 과정 토글
    # ============================================
    st.markdown("---")
    
    # 토글 상태 초기화
    if "show_trackb_calc" not in st.session_state:
        st.session_state.show_trackb_calc = False
    
    if st.button("▼ 이 결과는 어떻게 계산됐나요?" if not st.session_state.show_trackb_calc else "▲ 이 결과는 어떻게 계산됐나요?", key="toggle_trackb_calc"):
        st.session_state.show_trackb_calc = not st.session_state.show_trackb_calc
    
    if st.session_state.show_trackb_calc:
        st.markdown(
            """
<div style="background:#f1dddd; padding:24px; border-radius:12px; border:1px solid #e5e7eb; margin:16px 0;">
    <h3 style="color:#000000; margin-top:0;">① 집값 분포 가정</h3>
    
    <ul style="color:#000000;">
        <li>화곡동 빌라 매매 데이터를 기반으로 집값이 시간에 따라 변하는 확률 분포를 추정했어요.</li>
        <li>평균 성장률(μ)과 변동성(σ)을 사용했어요.</li>
        <li>최근 금리 수준을 반영해 μ를 보정했어요.</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">② 부도 확률 (PD)</h3>
    
    <ul style="color:#000000;">
        <li>계약 종료 시점에 집값 < 보증금 이 될 확률을 계산했어요.</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">③ 손실률 (LGD)</h3>
    
    <ul style="color:#000000;">
        <li>문제가 생길 경우, 경매 낙찰가가 감정가의 몇 % 수준인지 과거 경매 데이터를 통해 추정했어요.</li>
    </ul>
    
    <hr style="border:none; border-top:1px solid #d1d5db; margin:20px 0;">
    
    <h3 style="color:#000000;">④ 예상 손실 (EL)</h3>
    
    <ul style="color:#000000;">
        <li><strong>EL = PD × LGD × 보증금</strong></li>
        <li>"평균적으로 얼마를 잃을 수 있는지"를 의미해요.</li>
    </ul>
</div>
            """,
            unsafe_allow_html=True
        )

    if st.button("⬅︎ 요약으로 돌아가기", use_container_width=True):
        go("result")


# ✅ 맨 마지막에만 두기 (파일 제일 아래)
page = st.session_state.get("page", "input")

if page == "input":
    render_input()
    st.stop()
elif page == "result":
    render_result()
    st.stop()
elif page == "trackA":
    render_trackA()
    st.stop()
elif page == "trackB":
    render_trackB()
    st.stop()
else:
    st.session_state.page = "result"
    st.rerun()