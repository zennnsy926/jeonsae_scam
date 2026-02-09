import streamlit as st
import requests
import tracka_final as ta
import jeonse_ratio as jr
import numpy as np
import re

st.set_page_config(page_title="전세 위험도", layout="centered")

# === [ADD] 사분면 임계값(스크린샷 기준) ===
# 3단계 분류용 (안전/주의/고위험)
TRACKA_T1 = 0.56
TRACKA_T2 = 0.68

TRACKB_T1 = 0.210021
TRACKB_T2 = 0.472547

# 사분면 2단계 분류용 (저위험/고위험) - 더 보수적으로 설정
TRACKA_QUAD_THRESHOLD = 0.60  # 0.60 이상이면 사분면에서 고위험
TRACKB_QUAD_THRESHOLD = 0.40  # 0.40 이상이면 사분면에서 고위험

def go(page_name: str):
    st.session_state.page = page_name
    st.rerun()
    st.stop()


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
    """안전 / 주의 / 고위험"""
    if prob is None:
        return "N/A"
    if prob >= t2:
        return "고위험"
    if prob >= t1:
        return "주의"
    return "안전"


def bin_highlow(prob: float, t_high: float) -> str:
    """저위험 / 고위험 (2x2 사분면용)"""
    if prob is None:
        return "N/A"
    return "고위험" if prob >= t_high else "저위험"


def get_quadrant_case(a_bin: str, b_bin: str):
    """
    표 정의(너가 준 이미지 그대로)
      - A 고위험 & B 고위험 => Case 1: 의도적 사기 매물
      - A 저위험 & B 고위험 => Case 2: 시장 피해(역전세) 매물
      - A 저위험 & B 저위험 => Case 3: 안전 매물
      - A 고위험 & B 저위험 => Case 4: 특이 징후 매물
    """
    if a_bin == "고위험" and b_bin == "고위험":
        return ("Case 1", "의도적 사기 매물")
    if a_bin == "저위험" and b_bin == "고위험":
        return ("Case 2", "시장 피해(역전세) 매물")
    if a_bin == "저위험" and b_bin == "저위험":
        return ("Case 3", "안전 매물")
    if a_bin == "고위험" and b_bin == "저위험":
        return ("Case 4", "특이 징후 매물")
    return ("-", "-")


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
      :root{
        --bg:#f6f7f9;
        --text:#111827;
        --muted:#6b7280;
        --line:rgba(15,23,42,0.10);
        --shadow:0 6px 18px rgba(15, 23, 42, 0.08);
        --radius:18px;
        --btn:#163a66; /* 채도 낮은 진한 남색 */
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

      /* CTA 버튼: primary */
      button[kind="primary"]{
        width: 100% !important;
        min-width: 100% !important;
        height: 96px !important;
        border-radius: 26px !important;
        background: var(--btn) !important;
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 1000 !important;
        letter-spacing: -0.3px;
        border: 0 !important;
      }

      div.stButton, div[data-testid="stButton"]{ width: 100% !important; }
      div.stButton > button, div[data-testid="stButton"] > button{ width: 100% !important; }

      /* SLIDER: 트랙 두께 키우기 */
      div[data-baseweb="slider"]{ padding-top: 14px !important; padding-bottom: 10px !important; }
      div[data-baseweb="slider"] div[role="presentation"]{ height: 34px !important; }
      div[data-baseweb="slider"] div[role="presentation"] > div{ height: 34px !important; border-radius: 999px !important; }
      div[data-testid="stSlider"] div[role="presentation"]{ height: 34px !important; }
      div[data-testid="stSlider"] div[role="presentation"] > div{ height: 34px !important; border-radius: 999px !important; }

      /* thumb */
      div[data-baseweb="slider"] div[role="slider"]{
        width: 56px !important;
        height: 56px !important;
        border-radius: 999px !important;
        box-shadow: 0 14px 22px rgba(15,23,42,0.28) !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Views
# ----------------------------
def render_input():
    st.markdown('<div class="title">내가 선택한 이 집! 과연 안전할까?</div>', unsafe_allow_html=True)

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
            floor_options = ["반지하", "지상 1층"] + [f"지상 {i}층" for i in range(2, 41)] + ["옥탑"]
            FLOOR = st.selectbox("층", floor_options, index=1, key="floor_select")
        with c2:
            AREA_M2 = st.number_input("면적(㎡)", min_value=0.0, step=0.5, value=0.0, key="area_input")

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

    # 5) CTA
    clicked = st.button("이 조건으로 위험도 확인하기", type="primary", key="cta_to_result")

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
            "FLOOR_NUM": floor_to_num(FLOOR),
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



    # 6) 입력 기준(디자인용)
    with st.container(border=True):
        st.markdown('<div class="section-label">입력 기준 ❔</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sub" style="margin-top:8px; line-height:1.7;">
              • 이 집의 적정 매매가 V₀ = (추후 Track A 연결)<br>
              • (예시) 해당 지역 가격 평균 성장률 μ, 변동성 σ<br>
              • (예시) 해당 지역 경매 낙찰가율 평균 α<br><br>
              입력된 기준은 해당 지역 평균 통계를 바탕으로 산출됩니다.
            </div>
            """,
            unsafe_allow_html=True
        )



def render_result():
    st.markdown('<div class="title">요약</div>', unsafe_allow_html=True)

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

    # ---- 등급(3-bin) + 사분면용(2-bin) ----
    a_grade_3 = classify_3bin(float(probA), TRACKA_T1, TRACKA_T2)
    b_grade_3 = classify_3bin(float(probB), TRACKB_T1, TRACKB_T2)

    # 사분면은 별도 임계값 사용 (더 보수적)
    a_bin = bin_highlow(float(probA), TRACKA_QUAD_THRESHOLD)  # A: 0.60 이상이면 고위험
    b_bin = bin_highlow(float(probB), TRACKB_QUAD_THRESHOLD)  # B: 0.40 이상이면 고위험

    case_code, case_name = get_quadrant_case(a_bin, b_bin)

    # ---- 전세가율(있으면) ----
    # ---- 전세가율 (즉석 계산 or 저장값 사용) ----
    with st.container(border=True):
        st.markdown("### 전세가율")

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
            # (선택) 세션에 저장해두면 다른 페이지에서도 재사용 가능
            inputs["JEONSE_RATIO"] = float(ratio)
            st.session_state.inputs = inputs

    jeonse_ratio = inputs.get("JEONSE_RATIO", None)  # 너가 저장해두는 키에 맞춰서 사용
    # 만약 JEONSE_RATIO 없으면 v0 기반으로 즉석 계산하고 싶으면 여기서 계산해도 됨.

    # ---- UI 출력 ----
    st.subheader("최종 템플릿 위험 (사분면)")
    st.caption("Track A/B 임계값 기반으로 Case를 분류합니다.")

    st.markdown(
        """
        <style>
          .quad { width:100%; border-collapse:collapse; margin-top:8px; }
          .quad th, .quad td { border:1px solid rgba(15,23,42,0.12); padding:14px; text-align:center; font-weight:800; }
          .quad th { background: rgba(15,23,42,0.04); }
          .sel { outline: 3px solid rgba(22,58,102,0.7); background: rgba(22,58,102,0.06); }
          .small { font-size: 14px; font-weight: 700; color: #374151; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 선택 셀 class 결정
    sel_ah_bl = "sel" if (a_bin == "고위험" and b_bin == "저위험") else ""
    sel_ah_bh = "sel" if (a_bin == "고위험" and b_bin == "고위험") else ""
    sel_al_bl = "sel" if (a_bin == "저위험" and b_bin == "저위험") else ""
    sel_al_bh = "sel" if (a_bin == "저위험" and b_bin == "고위험") else ""

    st.markdown(
        f"""
        <table class="quad">
          <tr>
            <th></th>
            <th>Track B: 저위험</th>
            <th>Track B: 고위험</th>
          </tr>
          <tr>
            <th>Track A: 저위험</th>
            <td class="{sel_al_bl}">Case 3<br><span class="small">안전 매물</span></td>
            <td class="{sel_al_bh}">Case 2<br><span class="small">시장 피해(역전세) 매물</span></td>
          </tr>
          <tr>
            <th>Track A: 고위험</th>
            <td class="{sel_ah_bl}">Case 4<br><span class="small">특이 징후 매물</span></td>
            <td class="{sel_ah_bh}">Case 1<br><span class="small">의도적 사기 매물</span></td>
          </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    # 케이스별 메시지 결정
    if case_code == "Case 3":
        case_message = "✅ 완전히 안전합니다"
        message_color = "#10b981"  # 녹색
    elif case_code == "Case 4":
        case_message = "⚠️ 주의가 필요합니다"
        message_color = "#f59e0b"  # 주황색
    elif case_code == "Case 1":
        case_message = "🚨 사기일 가능성이 높습니다"
        message_color = "#ef4444"  # 빨간색
    elif case_code == "Case 2":
        case_message = "⚠️ 시장 상황에 따라 보증금을 다 돌려받지 못할 수 있습니다"
        message_color = "#f59e0b"  # 주황색
    else:
        case_message = "분석 결과를 확인하세요"
        message_color = "#6b7280"

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px;">
                <div style="font-size:28px; font-weight:900; color:{message_color}; margin-bottom:10px;">
                    {case_code}: {case_name}
                </div>
                <div style="font-size:18px; font-weight:700; color:#374151;">
                    {case_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 상세 페이지 이동 버튼 2개
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Track A 값 확인하러 가기", key="goA_from_result"):
            go("trackA")
    with c2:
        if st.button("Track B 값 확인하러 가기", key="goB_from_result"):
            go("trackB")

    st.write("")
    if st.button("⬅︎ 입력 화면으로 돌아가기", use_container_width=True, key="back_to_input"):
        go("input")


def render_trackA():
    import numpy as np
    import tracka_final as ta

    def _is_nan(v):
        try:
            return v is None or (isinstance(v, float) and np.isnan(v))
        except Exception:
            return v is None

    st.markdown("## Track A")
    st.caption("적정 매매가(V₀) 및 Track A 세부 계산 결과입니다.")

    inputs = st.session_state.get("inputs", {})

    # 입력값 표시 (예쁜 카드 형식)
    with st.container(border=True):
        st.markdown("### 현재 입력값")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📍 주소**: {inputs.get('ROAD_ADDR', 'N/A')}")
            st.markdown(f"**🏢 층**: {inputs.get('FLOOR', 'N/A')}")
            st.markdown(f"**📐 면적**: {inputs.get('AREA_M2', 0):.2f}㎡")
        with col2:
            st.markdown(f"**💰 보증금**: {inputs.get('DEPOSIT', 0):,}만원")
            st.markdown(f"**📅 계약기간**: {inputs.get('CONTRACT_YEARS', 0)}년")
            st.markdown(f"**🔢 지번**: {inputs.get('JIBUN', 'N/A')}")

    # 이미 계산된 값 재사용
    if "TRACKA_RESULT" in inputs and "TRACKA_COMMENTS" in inputs and (not _is_nan(inputs.get("V0", None))):
        resA = inputs["TRACKA_RESULT"]
        commentsA = inputs["TRACKA_COMMENTS"]
    else:
        # 없으면 계산
        try:
            with st.spinner("Track A 계산 중..."):
                resA, commentsA = ta.predict_final(
                    jibun=inputs["JIBUN"],
                    area_m2=float(inputs["AREA_M2"]),
                    floor=int(inputs.get("FLOOR_NUM", 1)),   # 없으면 1층 fallback
                    deposit=int(inputs["DEPOSIT"]),
                )

            # ✅ 여기서 “한 번만” 저장 (핵심)
            inputs["TRACKA_RESULT"] = resA
            inputs["TRACKA_COMMENTS"] = commentsA

            # ✅ V0 확정 저장 (TrackA/요약/TrackB/전세가율이 이걸 씀)
            v0 = resA.get("V0", None)
            inputs["V0"] = float(resA.get("V0", np.nan))

            st.session_state.inputs = inputs

        except Exception as e:
            st.error("Track A 계산 중 오류가 발생했어요.")
            st.exception(e)
            return

    # Track A 결과 카드
    prob_value = resA.get("prob", 0)
    grade_value = resA.get("grade", "N/A")
    v0_value = resA.get("V0", 0)
    
    # 위험 등급에 따른 색상 결정
    if grade_value == "고위험":
        grade_color = "#ef4444"  # 빨간색
        grade_bg = "#fee2e2"
        grade_icon = "🚨"
    elif grade_value == "주의":
        grade_color = "#f59e0b"  # 주황색
        grade_bg = "#fef3c7"
        grade_icon = "⚠️"
    else:  # 안전
        grade_color = "#10b981"  # 녹색
        grade_bg = "#d1fae5"
        grade_icon = "✅"
    
    with st.container(border=True):
        st.markdown("### Track A 결과")
        
        # 등급 박스 (강조)
        st.markdown(
            f"""
            <div style="background:{grade_bg}; padding:20px; border-radius:12px; margin-bottom:16px; border-left:6px solid {grade_color};">
                <div style="font-size:16px; font-weight:700; color:#374151; margin-bottom:6px;">위험 등급</div>
                <div style="font-size:36px; font-weight:900; color:{grade_color};">
                    {grade_icon} {grade_value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 수치 카드
        col1, col2 = st.columns(2)
        with col1:
            st.metric("위험 확률", f"{prob_value:.2%}")
        with col2:
            st.metric("적정 매매가(V₀)", f"{v0_value:,.0f}만원")

    # 해석 카드
    with st.container(border=True):
        st.markdown("### 해석")
        for c in commentsA:
            st.markdown(f"• {c}")

    if st.button("⬅︎ 요약으로 돌아가기", use_container_width=True):
        go("result")



def render_trackB():
    import numpy as np
    import pandas as pd
    import trackb_final as tb

    st.markdown("## Track B")
    st.caption("GBM 기반 PD/EL 및 전세가율을 계산합니다.")

    inputs = st.session_state.get("inputs", {})

    # 입력값 표시 (예쁜 카드 형식)
    with st.container(border=True):
        st.markdown("### 현재 입력값")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📍 주소**: {inputs.get('ROAD_ADDR', 'N/A')}")
            st.markdown(f"**🏢 층**: {inputs.get('FLOOR', 'N/A')}")
            st.markdown(f"**📐 면적**: {inputs.get('AREA_M2', 0):.2f}㎡")
        with col2:
            st.markdown(f"**💰 보증금**: {inputs.get('DEPOSIT', 0):,}만원")
            st.markdown(f"**📅 계약기간**: {inputs.get('CONTRACT_YEARS', 0)}년")
            st.markdown(f"**🔢 지번**: {inputs.get('JIBUN', 'N/A')}")

    # Track A 결과(V0) 없으면 안내
    V0 = inputs.get("V0", None)
    if V0 is None or (isinstance(V0, float) and np.isnan(V0)):
        st.warning("Track B를 계산하려면 Track A에서 산출된 적정 매매가가 필요해요. Track A를 먼저 실행해 주세요.")
        if st.button("⬅︎ 요약으로 돌아가기", use_container_width=True):
            go("result")
        return

    # Track B 입력값 구성 (단위: 만원)
    B = float(inputs["DEPOSIT"])
    T = float(inputs["CONTRACT_YEARS"])

    df_in = pd.DataFrame([{
        "hedonic_price": float(V0),  # V0
        "deposit": B,               # 보증금
        "term": T,                  # 계약기간(년)
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

            # B* (적정보증금 상한) : base 시나리오 기준, mu 보정 전/후 범위
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

    # === 결과 요약 섹션 ===
    current_deposit = float(inputs["DEPOSIT"])
    pd_value = row['PD_base']
    
    # 위험도 판단
    is_high_pd = pd_value >= 0.30  # PD 30% 이상이면 고위험
    is_deposit_over = current_deposit > b_after  # 보증금이 적정 범위 초과
    
    if is_high_pd and is_deposit_over:
        risk_level = "⛔ 고위험"
        risk_bg = "#fee2e2"
        risk_color = "#991b1b"
        risk_message = "현재 조건에서는 보증금을 온전히 돌려받지 못할 가능성이 매우 높아요."
        risk_items = [
            "보증금이 적정 범위를 초과했어요",
            "손실 확률이 높은 수준이에요",
            "계약 재검토를 권장합니다"
        ]
    elif is_high_pd or is_deposit_over:
        risk_level = "⚠️ 주의"
        risk_bg = "#fef3c7"
        risk_color = "#92400e"
        risk_message = "현재 조건에서는 보증금을 온전히 돌려받지 못할 가능성이 비교적 높아요."
        risk_items = [
            "보증금이 적정 범위에 근접했거나 초과했어요" if is_deposit_over else "손실 확률이 다소 높아요",
            "집값 변동성을 주의깊게 모니터링하세요",
            "경매 시 낙찰가가 감정가보다 낮게 형성될 수 있어요"
        ]
    else:
        risk_level = "✅ 안전"
        risk_bg = "#d1fae5"
        risk_color = "#065f46"
        risk_message = "현재 조건에서는 보증금을 돌려받을 가능성이 높아요."
        risk_items = [
            "보증금이 적정 범위 이내에요",
            "손실 확률이 낮은 수준이에요",
            "비교적 안전한 조건입니다"
        ]
    
    with st.container(border=True):
        st.markdown("## 이 전세, 돈을 잃을 확률은 얼마인가요?")
        
        # 위험 경고
        risk_items_html = "".join([f"<li>{item}</li>" for item in risk_items])
        st.markdown(
            f"""
            <div style="background:{risk_bg}; padding:16px; border-radius:12px; margin:12px 0;">
                <div style="font-size:18px; font-weight:800; color:{risk_color}; margin-bottom:8px;">{risk_level}</div>
                <div style="color:{risk_color}; font-weight:600;">{risk_message}</div>
                <div style="margin-top:12px; color:{risk_color}; font-weight:600;">주요 요소 요약</div>
                <ul style="margin:8px 0; padding-left:20px; color:{risk_color};">
                    {risk_items_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # === 주요 수치 카드 ===
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❓ 보증금을 못 돌려받을 확률")
            st.markdown(f"<div style='font-size:42px; font-weight:900; color:#163a66;'>{row['PD_base']:.1%}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ❓ 평균적으로 잃을 수 있는 금액")
            st.markdown(f"<div style='font-size:32px; font-weight:900; color:#163a66;'>예상 평균 손실액: 약 {row['EL_base']:,.0f}만원</div>", unsafe_allow_html=True)

    # === 적정 보증금 범위 ===
    current_deposit = float(inputs["DEPOSIT"])
    
    # 현재 보증금과 적정 범위 비교
    if current_deposit <= b_after:
        safety_status = "안전"
        safety_color = "#10b981"  # 녹색
        safety_message = f"현재 보증금({current_deposit:,.0f}만원)은 적정 범위 이하로 안전합니다."
    else:
        safety_status = "위험"
        safety_color = "#ef4444"  # 빨간색
        over_amount = current_deposit - b_after
        safety_message = f"현재 보증금({current_deposit:,.0f}만원)이 적정 범위를 {over_amount:,.0f}만원 초과합니다. 위험할 수 있습니다."
    
    with st.container(border=True):
        st.markdown("### ❓ 적정 보증금 범위 분석")
        st.markdown(
            f"""
            <div style="border:2px solid #e5e7eb; padding:16px; border-radius:8px; background:#f9fafb;">
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

    # === 위험 민감도 시나리오 ===
    with st.container(border=True):
        st.markdown("## 위험 민감도별 시나리오")
        st.markdown("시장 상황이 나빠질수록 손실 위험이 얼마나 커지는지 보여줘요.")
        
        # 테이블 생성
        st.markdown(
            """
            <table style="width:100%; border-collapse:collapse; margin-top:16px;">
                <thead>
                    <tr style="background:#f3f4f6;">
                        <th style="padding:12px; border:1px solid #e5e7eb; font-weight:800;">시장 시나리오</th>
                        <th style="padding:12px; border:1px solid #e5e7eb; font-weight:800;">예상 평균 손실</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding:12px; border:1px solid #e5e7eb;">정상</td>
                        <td style="padding:12px; border:1px solid #e5e7eb; font-weight:800;">약 {EL_BASE}만 원</td>
                    </tr>
                    <tr>
                        <td style="padding:12px; border:1px solid #e5e7eb;">-10% 하락</td>
                        <td style="padding:12px; border:1px solid #e5e7eb; font-weight:800;">약 {EL_10}만 원</td>
                    </tr>
                    <tr>
                        <td style="padding:12px; border:1px solid #e5e7eb;">-20% 하락</td>
                        <td style="padding:12px; border:1px solid #e5e7eb; font-weight:800;">약 {EL_20}만 원</td>
                    </tr>
                </tbody>
            </table>
            """.replace("{EL_BASE}", f"{row['EL_base']:,.0f}")
               .replace("{EL_10}", f"{row.get('EL_stress10', 0):,.0f}")
               .replace("{EL_20}", f"{row['EL_stress20']:,.0f}"),
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