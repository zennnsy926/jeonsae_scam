import pickle
import warnings
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# ==========================================
# PNU 변환 함수
# ==========================================
def ltno_to_pnu(ltno, dong_code="1150010300"):
    """지번 → PNU 변환 (화곡동 기준)"""
    if pd.isna(ltno) or str(ltno).lower() == "nan":
        return None

    ltno = str(ltno).strip()

    if "-" in ltno:
        main, sub = ltno.split("-")
    else:
        main, sub = ltno, "0"

    main = str(int(float(main))).zfill(4)
    sub = str(int(float(sub))).zfill(4)
    land_type = "1"  # 일반 토지

    return dong_code + land_type + main + sub


# ==========================================
# Assets (데이터/모델/패키지) - Streamlit rerun 대비 캐시
# ==========================================
@lru_cache(maxsize=1)
def load_assets():
    """
    Streamlit에서 import 후 여러 번 호출되어도
    데이터/모델을 프로세스당 1번만 로드하도록 캐싱.
    """
    df_trade = pd.read_csv(DATA_DIR / "MD1_final.csv", dtype={"PNU": str})
    df_lease = pd.read_csv(DATA_DIR / "MD2_final.csv", dtype={"PNU": str})
    pnu_location = pd.read_csv(DATA_DIR / "PNU_location.csv", dtype={"PNU": str})

    # hedonic_model.pkl: dict 형태(model, selected_features)
    with open(MODELS_DIR / "hedonic_model.pkl", "rb") as f:
        hedonic_pkg = pickle.load(f)

    # hwagok_auction_risk_model.pkl: dict 형태(model, bins_config, woe_maps, features)
    auction_pkg = joblib.load(MODELS_DIR / "hwagok_auction_risk_model.pkl")

    # 전체 의심사례(분모) 계산: 경매_4년이내 == 1인 건수
    # (원하는 분모 정의가 따로 있으면 여기만 바꾸면 됨)
    if "경매_4년이내" in df_lease.columns:
        total_suspected = int(pd.to_numeric(df_lease["경매_4년이내"], errors="coerce").fillna(0).sum())
    else:
        total_suspected = 0

    return df_trade, df_lease, pnu_location, hedonic_pkg, auction_pkg, total_suspected


# ==========================================
# 1단계: 헤도닉 예측 (매매 적정가)
# ==========================================
def predict_hedonic_price(jibun, area_m2, floor, df_trade, pnu_location, model_package):
    """헤도닉 모델로 매매 적정가 예측 (단위: '만원'이라고 가정)"""

    pnu = ltno_to_pnu(jibun)
    if pnu is None:
        raise ValueError(f"유효하지 않은 지번: {jibun}")

    pnu = str(pnu)
    matching = df_trade[df_trade["PNU"] == pnu]

    if len(matching) == 0:
        raise ValueError(f"PNU {pnu}에 해당하는 매매 이력이 없습니다")

    latest = matching.sort_values("계약일", ascending=False).iloc[0]

    location_matching = pnu_location[pnu_location["PNU"] == pnu]
    if len(location_matching) == 0:
        raise ValueError(f"PNU {pnu}에 해당하는 위경도 정보가 없습니다")

    lat = location_matching["위도"].iloc[0]
    lon = location_matching["경도"].iloc[0]

    area_pyeong = float(area_m2) / 3.3058

    features = {
        "전용면적_평": area_pyeong,
        "층": int(floor),
        "건축연령": latest["건축연령"] if pd.notna(latest.get("건축연령", np.nan)) else 0,
        "건축연령_sq": (latest["건축연령"] ** 2) if pd.notna(latest.get("건축연령", np.nan)) else 0,
        "area_floor_inter": area_pyeong * int(floor),
        "관내": latest["관내"] if pd.notna(latest.get("관내", np.nan)) else 0,
        "전월세_평균_보증금(만원)": latest["전월세_평균_보증금(만원)"] if pd.notna(latest.get("전월세_평균_보증금(만원)", np.nan)) else 0,
        "기준금리(연%)": 2.5,
        "전월세_평균_월세(만원)": latest["전월세_평균_월세(만원)"] if pd.notna(latest.get("전월세_평균_월세(만원)", np.nan)) else 0,
        "전월세_건수": latest["전월세_건수"] if pd.notna(latest.get("전월세_건수", np.nan)) else 0,
        "공원_최단거리": latest["공원_최단거리"] if pd.notna(latest.get("공원_최단거리", np.nan)) else 0,
        "교육_최단거리": latest["교육_최단거리"] if pd.notna(latest.get("교육_최단거리", np.nan)) else 0,
        "유통_최단거리": latest["유통_최단거리"] if pd.notna(latest.get("유통_최단거리", np.nan)) else 0,
        "매수자_법인": latest["매수자_법인"] if pd.notna(latest.get("매수자_법인", np.nan)) else 0,
        "매도자_개인": latest["매도자_개인"] if pd.notna(latest.get("매도자_개인", np.nan)) else 0,
        "매도자_M": latest["매도자_M"] if pd.notna(latest.get("매도자_M", np.nan)) else 0,
        "거래유형_직거래": latest["거래유형_직거래"] if pd.notna(latest.get("거래유형_직거래", np.nan)) else 0,
        "age_buycorp_inter": (latest["건축연령"] * latest["매수자_법인"])
        if (pd.notna(latest.get("건축연령", np.nan)) and pd.notna(latest.get("매수자_법인", np.nan)))
        else 0,
    }

    model = model_package["model"]
    selected_features = model_package["selected_features"]

    input_df = pd.DataFrame([features])[selected_features]
    X_new = sm.add_constant(input_df, has_constant="add")

    ln_price = model.predict(X_new)[0]
    actual_price = float(np.exp(ln_price))  # 단위가 '만원'이라고 가정

    return actual_price, float(lat), float(lon)


# ==========================================
# 2단계: 로지스틱 회귀 파생변수 생성
# ==========================================
def create_logistic_features(df_jeonse, deposit, hedonic_price, user_lat, user_lon):
    """로지스틱 회귀용 파생변수 생성"""

    # 단위 통일 가정: deposit, hedonic_price 모두 '만원'
    effective_LTV = (float(deposit) / float(hedonic_price)) * 100 if hedonic_price else 0.0
    deposit_overhang = float(deposit) - float(hedonic_price)

    df_clean = df_jeonse.dropna(subset=["경도", "위도", "Residual"]).copy()

    coords = df_clean[["경도", "위도"]].values
    user_coord = np.array([[float(user_lon), float(user_lat)]])

    # 대략적인 km 스케일링(서울 근처 근사)
    coords_scaled = coords.copy()
    coords_scaled[:, 0] *= 88
    coords_scaled[:, 1] *= 111

    user_coord_scaled = user_coord.copy()
    user_coord_scaled[:, 0] *= 88
    user_coord_scaled[:, 1] *= 111

    distances = cdist(user_coord_scaled, coords_scaled)[0]

    threshold_km = 1
    neighbors = distances < threshold_km
    auction_flags = df_clean["경매_4년이내"].values
    nearby_auction = int((neighbors & (auction_flags == 1)).sum())

    nearest_idx = int(np.argmin(distances))
    local_morans_i = float(df_clean.iloc[nearest_idx]["local_morans_i"])

    features = {
        "effective_LTV": float(effective_LTV),
        "deposit_overhang": float(deposit_overhang),
        "nearby_auction_1km": nearby_auction,
        "local_morans_i": local_morans_i,
    }

    return features


# ==========================================
# 3단계: 로지스틱 회귀 예측 (WoE + 모델)
# ==========================================
def predict_auction_risk(new_data_dict, auction_pkg):
    """경매 위험 확률 예측"""

    model = auction_pkg["model"]
    bins_config = auction_pkg["bins_config"]
    woe_maps = auction_pkg["woe_maps"]
    features = auction_pkg["features"]

    # 입력 데이터를 WoE로 변환
    woe_vector = []
    for col in features:
        val = new_data_dict.get(col, 0)
        bin_idx = pd.cut([val], bins=bins_config[col], labels=False, include_lowest=True)[0]
        woe_val = woe_maps[col].get(bin_idx, 0)
        woe_vector.append(woe_val)

    prob = float(model.predict_proba([woe_vector])[0, 1])
    grade = "고위험" if prob >= 0.63 else ("주의" if prob >= 0.53 else "안전")

    return {"prob": round(prob, 4), "grade": grade}


# ==========================================
# 설명 문장 생성
# ==========================================
def generate_fact_comments(logistic_features, total_suspected: int):
    comments = []

    ltv = float(logistic_features.get("effective_LTV", 0))
    nearby = int(logistic_features.get("nearby_auction_1km", 0))

    comments.append(
        f"전세금이 매매 적정가의 {ltv:.2f}%를 차지해요. 집주인의 실소유 지분이 {max(0.0, 100-ltv):.2f}% 정도로 추정돼요."
    )

    if total_suspected and total_suspected > 0:
        pct = (nearby / total_suspected) * 100
        comments.append(
            f"주변 1km 내에 전세사기 의심 사례로 분류된 경매가 최근 4년간 {nearby}건 있어요. "
            f"(전체 의심 사례 {total_suspected}건 중 약 {pct:.2f}%)"
        )
    else:
        comments.append(
            f"주변 1km 내 전세사기 의심 사례(경매_4년이내)가 최근 4년간 {nearby}건 있어요."
        )

    return comments


# ==========================================
# 통합 예측 함수 (Streamlit에서 이거만 호출하면 됨)
# ==========================================
def predict_final(jibun, area_m2, floor, deposit):
    """
    전체 파이프라인: 사용자 입력 → 경매 위험 확률
    반환:
      result: {'prob': 0~1, 'grade': '안전/주의/고위험'}
      comments: 설명 문장 리스트
    """
    df_trade, df_lease, pnu_location, hedonic_pkg, auction_pkg, total_suspected = load_assets()

    hedonic_price, lat, lon = predict_hedonic_price(
        jibun=jibun,
        area_m2=area_m2,
        floor=floor,
        df_trade=df_trade,
        pnu_location=pnu_location,
        model_package=hedonic_pkg,
    )

    logistic_features = create_logistic_features(
        df_jeonse=df_lease,
        deposit=deposit,
        hedonic_price=hedonic_price,
        user_lat=lat,
        user_lon=lon,
    )

    result = predict_auction_risk(logistic_features, auction_pkg)

    # ✅ V0(적정 매매가)도 같이 담아서 스트림릿/TrackB에서 쓰게 하기
    result["V0"] = float(hedonic_price) if hedonic_price is not None else None

    comment = generate_fact_comments(logistic_features, total_suspected)

    return result, comment



# ==========================================
# 로컬 테스트용 실행부
# ==========================================
if __name__ == "__main__":
    # 예시(테스트)
    result, comments = predict_final(
        jibun="1036-01",
        area_m2=27.79,
        floor=6,
        deposit=17000,
    )

    print(f"경매 위험 확률: {result['prob']*100:.2f}%")
    print(f"위험 등급: {result['grade']}")
    for i, c in enumerate(comments, 1):
        print(f"{i}. {c}")