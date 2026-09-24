# =================배포 추론====================
# MS data school 4기 1차 프로젝트
# 1팀(팀명: eve) 모델 배포 추론 코드

### 작업환경 : Azure ML Studio > Notebooks
### subscription_id  : "<YOUR_SUBSCRIPTION_ID>"
### resource_group   : "<YOUR_RESOURCE_GROUP>"
### workspace_name   : "<YOUR_WORKSPACE_NAME>"
### endpoint_name    : "<YOUR_ENDPOINT_NAME>"
### deployment       : "<YOUR_DEPLOYMENT_NAME>"
### =====================================

import json
import os
import joblib
import logging
import numpy as np
import pandas as pd

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType


# ============================================================
# [1] Global variables
# ============================================================

model = None
classes_ = None
danger_label = None
danger_threshold = None
danger_idx = None

MODEL_FEATURE_COLS = [
    # 배터리 전압
    "voltage",

    # 배터리 전류
    "current",

    # 배터리 온도
    "battery_temp",

    # 주변 온도
    "ambient_temp",

    # 전압 변화량
    "delta_v",

    # 전류 변화량
    "delta_i",

    # 온도차 = 배터리 온도 - 주변 온도
    "temp_diff",

    # 전류기반 열 스트레스 지수
    "joule_heating_stress",

    # 70분 롤링 절대 전력(배터리 부하) 평균
    "rolling_abs_power_70min",

    # 온도차 지수
    "thermal_stress",

    # 70분 롤링 온도차 평균
    "temp_diff_mean"
]


# ============================================================
# [1-1] Input / Output schema
# ============================================================


sample_input = StandardPythonParameterType({
    "Inputs": {
        "WebServiceInput0": [{
            "vehicle_id": "EV001",
            "model_name": "IONIQ5",
            "received_at": "2026-05-27T10:00:00Z",
            "battery_voltage": 350.0,
            "battery_current": 20.0,
            "temperature": 28.0,
            "ambient_temp": 22.0,
            "delta_i": 0.2,
            "delta_v": 0.5,
            "joule_heating_stress": 0.3,
            "latitude": 35.1796,
            "longitude": 129.0756,
            "current_region_id": 101,
            "region_name": "Busan",
            "is_active": 1,
            "alert_type": "NONE"
        }]
    }
})



outputs = StandardPythonParameterType([{
    "vehicle_id": "EV001",
    "model_name": "IONIQ5",
    "received_at": "2026-05-27T10:00:00Z",
    "battery_voltage": 350.0,
    "battery_current": 20.0,
    "temperature": 28.0,
    "ambient_temp": 22.0,
    "delta_i": 0.2,
    "delta_v": 0.5,
    "joule_heating_stress": 0.3,
    "latitude": 35.1796,
    "longitude": 129.0756,
    "current_region_id": 101,
    "region_name": "Busan",
    "is_active": 1,
    "alert_type": "NONE",
    "current_bsi": 1.12,
    "status": "NORMAL"
}])


# ============================================================
# [2] Feature engineering
# ============================================================

def make_features(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --------------------------------------------------------
    # [수정/보강] 숫자 컬럼 타입 안정화
    # - ASA/IoT Hub에서 숫자가 문자열로 들어오는 경우를 대비
    # - feature 계산 전에 숫자로 변환
    # --------------------------------------------------------
    numeric_cols = [
        "battery_voltage",
        "battery_current",
        "temperature",
        "ambient_temp",
        "delta_i",
        "delta_v",
        "joule_heating_stress",
        "latitude",
        "longitude",
        "current_region_id",
        "is_active"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 데이터 측정 시각 datetime 타입으로 변환       
    df["received_at"] = pd.to_datetime(df["received_at"])

    # 차량id + 측정 시각을 인덱스로 취급하여 정렬
    df = df.sort_values(["vehicle_id", "received_at"]).reset_index(drop=True)

    # 온도차 계산 = 배터리 온도 - 주변 온도 
    df["temp_diff"] = df["temperature"] - df["ambient_temp"]

    # 배터리 전력 = 배터리 전압 * 배터리 전류
    df["power"] = df["battery_voltage"] * df["battery_current"]

    # 배터리 전력 절대값
    df["abs_power"] = df["power"].abs()

    # 온도차 지수 = 온도차의 절대값
    df["thermal_stress"] = df["temp_diff"].abs()

    result_list = []

    for vid, g in df.groupby("vehicle_id", group_keys=False):
        g = g.sort_values("received_at").copy()
        g = g.set_index("received_at")

        # 70분 롤링 절대 전력(배터리 부하) 평균
        g["rolling_abs_power_70min"] = (
            g["abs_power"]
            .rolling("70min", min_periods=1)
            .mean()
        )

        # 70분 롤링 온도차 평균    
        g["temp_diff_mean"] = (
            g["temp_diff"]
            .rolling("70min", min_periods=1)
            .mean()
        )

        # ------------------------------------------------
        # 3분 롤링 평균과 표준편차로 z-score 계산
        #  std가 0이 되는 경우를 대비하여 inf 처리 및 fillna(0) 추가
        #  inf 처리 추가
        # ------------------------------------------------
        # 전류변화를 직전 3분 동안 z-정규화
        deli = g["delta_i"].abs()
        deli_mean = deli.rolling("3min", min_periods=1).mean()
        deli_std = deli.rolling("3min", min_periods=1).std()
        g["z_delta_i"] = (
            (deli - deli_mean) / deli_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

         # 전압변화를 직전 3분 동안 z-정규화
        delv = g["delta_v"].abs()
        delv_mean = delv.rolling("3min", min_periods=1).mean()
        delv_std = delv.rolling("3min", min_periods=1).std()
        g["z_delta_v"] = (
            (delv - delv_mean) / delv_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # 전류기반 열 스트레스를 직전 3분 동안 z-정규화
        jhs = g["joule_heating_stress"]
        jhs_mean = jhs.rolling("3min", min_periods=1).mean()
        jhs_std = jhs.rolling("3min", min_periods=1).std()
        g["z_joule_heating_stress"] = (
            (jhs - jhs_mean) / jhs_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # 온도차 지수를 직전 3분 동안 z-정규화
        ts = g["thermal_stress"].abs()
        ts_mean = ts.rolling("3min", min_periods=1).mean()
        ts_std = ts.rolling("3min", min_periods=1).std()
        g["z_thermal_stress"] = (
            (ts - ts_mean) / ts_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # 배터리 전류를 직전 3분 동안 z-정규화
        c = g["battery_current"].abs()
        c_mean = c.rolling("3min", min_periods=1).mean()
        c_std = c.rolling("3min", min_periods=1).std()
        g["z_battery_current"] = (
            (c - c_mean) / c_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # 배터리 전압을 직전 3분 동안 z-정규화
        v = g["battery_voltage"].abs()
        v_mean = v.rolling("3min", min_periods=1).mean()
        v_std = v.rolling("3min", min_periods=1).std()
        g["z_battery_voltage"] = (
            (v - v_mean) / v_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # bsi 계산
        g["bsi"] = (
            0.4830 * g["z_delta_i"].abs()
            + 0.2218 * g["z_delta_v"].abs()
            + 0.1027 * g["z_thermal_stress"].abs()
            + 0.0992 * g["z_battery_current"].abs()
            + 0.0933 * g["z_battery_voltage"].abs()
        )

        # bsi를 직전 3분 동안 z-정규화
        bsi_abs = g["bsi"].abs()
        bsi_mean = bsi_abs.rolling("3min", min_periods=1).mean()
        bsi_std = bsi_abs.rolling("3min", min_periods=1).std()
        g["z_bsi"] = (
            (bsi_abs - bsi_mean) / bsi_std
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # bsi_label 계산
        g["bsi_label"] = np.select(
            [
                g["z_bsi"] < 2,
                (g["z_bsi"] >= 2) & (g["z_bsi"] < 3),
                g["z_bsi"] >= 3
            ],
            [0, 1, 2],
            default=0
        )

        g = g.reset_index()
        result_list.append(g)

    feature_df = pd.concat(result_list, axis=0).reset_index(drop=True)
    return feature_df


# ============================================================
# [3] Model init
# ============================================================

def init():
    global model, classes_, danger_label, danger_threshold, danger_idx

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # --------------------------------------------------------
    # [수정/보강] model_dir None 체크
    # - Azure ML 모델 경로 환경변수가 없는 경우 init 단계에서 명확히 에러를 냄
    # --------------------------------------------------------
    if model_dir is None:
        raise ValueError("AZUREML_MODEL_DIR is None. Azure ML 모델 경로를 확인해야 합니다.")

    artifact_dir = os.path.join(model_dir, "ev_lgbm_inference_artifact")

    model_path = os.path.join(artifact_dir, "model.pkl")
    class_info_path = os.path.join(artifact_dir, "class_info.json")
    threshold_path = os.path.join(artifact_dir, "threshold.json")

    # --------------------------------------------------------
    # [수정/보강] 모델 artifact 경로 로그 추가
    # - 배포 후 model.pkl/class_info.json/threshold.json 경로 문제를 확인하기 위함
    # --------------------------------------------------------
    logging.info(f"=== AZUREML_MODEL_DIR: {model_dir}")
    logging.info(f"=== artifact_dir: {artifact_dir}")
    logging.info(f"=== model_path exists: {os.path.exists(model_path)}")
    logging.info(f"=== class_info_path exists: {os.path.exists(class_info_path)}")
    logging.info(f"=== threshold_path exists: {os.path.exists(threshold_path)}")

    model = joblib.load(model_path)

    with open(class_info_path, "r", encoding="utf-8") as f:
        classes_ = np.array(json.load(f)["classes"])

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    danger_label = threshold_info["danger_label"]
    danger_threshold = threshold_info["danger_threshold"]

    danger_match = np.where(classes_ == danger_label)[0]

    # --------------------------------------------------------
    # [수정/보강] danger_label 존재 여부 검증
    # - threshold.json의 danger_label이 class_info.json의 classes에 없으면 명확히 에러 처리
    # --------------------------------------------------------
    if len(danger_match) == 0:
        raise ValueError(
            f"danger_label {danger_label} not found in classes_ {classes_.tolist()}"
        )

    danger_idx = danger_match[0]


# ============================================================
# [4] Endpoint run
# ============================================================
@input_schema("Inputs", sample_input)
@output_schema(outputs)
def run(Inputs):
    try:
        raw_data = Inputs

        # ----------------------------------------------------
        # 입력 로그 추가
        # - ASA/Azure ML에서 실제로 어떤 형태로 들어오는지 확인하기 위함
        # ----------------------------------------------------
        logging.info(f"=== RECEIVED RAW TYPE: {type(raw_data)}")
        logging.info(f"=== RECEIVED RAW DATA: {str(raw_data)[:1500]}")

        # ----------------------------------------------------
        # 문자열 JSON 처리
        # - 입력이 문자열 JSON으로 들어오는 경우를 대비
        # ----------------------------------------------------
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        # ----------------------------------------------------
        # DataFrame 입력 처리
        # - inference_schema가 DataFrame으로 넘기는 경우를 대비
        # ----------------------------------------------------
        if isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()

        else:
            # ------------------------------------------------
            # {"Inputs": {"WebServiceInput0": [...]}} 형태 처리
            # ------------------------------------------------
            if isinstance(raw_data, dict) and "Inputs" in raw_data:
                input_payload = raw_data["Inputs"]

                if isinstance(input_payload, dict) and "WebServiceInput0" in input_payload:
                    records = input_payload["WebServiceInput0"]
                else:
                    raise ValueError(
                        f"Inputs 안에 WebServiceInput0 없음. input_payload={input_payload}"
                    )

            # ------------------------------------------------
            # {"WebServiceInput0": [...]} 형태 처리
            # ------------------------------------------------
            elif isinstance(raw_data, dict) and "WebServiceInput0" in raw_data:
                records = raw_data["WebServiceInput0"]

            # ------------------------------------------------
            # [ {...}, {...} ] 배열 형태 처리
            # ------------------------------------------------
            elif isinstance(raw_data, list):
                records = raw_data

            # ------------------------------------------------
            # 단일 record dict 형태 처리
            # ------------------------------------------------
            elif isinstance(raw_data, dict):
                records = [raw_data]

            else:
                raise ValueError(f"Unsupported data type: {type(raw_data)}")

            # ------------------------------------------------
            # records가 문자열 JSON으로 들어온 경우 처리
            # ------------------------------------------------
            if isinstance(records, str):
                records = json.loads(records)

            # ------------------------------------------------
            # records가 단일 dict면 list로 변환
            # ------------------------------------------------
            if isinstance(records, dict):
                records = [records]

            df = pd.DataFrame(records)

        df.columns = [str(c).strip() for c in df.columns]

        # ----------------------------------------------------
        # parsing 결과 로그
        # ----------------------------------------------------
        logging.info(f"=== PARSED DF SHAPE: {df.shape}")
        logging.info(f"=== PARSED DF COLUMNS: {df.columns.tolist()}")
        logging.info(f"=== PARSED DF HEAD: {df.head(3).to_dict(orient='records')}")

        # 입력받는 컬럼
        required_cols = [
            "vehicle_id",
            "model_name",
            "received_at",
            "battery_voltage",
            "battery_current",
            "temperature",
            "ambient_temp",
            "delta_i",
            "delta_v",
            "joule_heating_stress",
            "latitude",
            "longitude",
            "current_region_id",
            "region_name",
            "is_active",
            "alert_type"
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        feature_df = make_features(df).copy()

        # 이름 매칭
        model_input_df = feature_df.rename(columns={
            "battery_voltage": "voltage",
            "battery_current": "current",
            "temperature": "battery_temp"
        })

        X_pred = model_input_df[MODEL_FEATURE_COLS].copy()

        for col in X_pred.columns:
            X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce")

        X_pred = X_pred.fillna(0)

        # ----------------------------------------------------
        # 모델 입력 로그
        # ----------------------------------------------------
        logging.info(f"=== MODEL INPUT SHAPE: {X_pred.shape}")
        logging.info(f"=== MODEL INPUT HEAD: {X_pred.head(3).to_dict(orient='records')}")

        proba = model.predict_proba(X_pred)

        preds = []

        for p in proba:
            if p[danger_idx] >= danger_threshold:
                preds.append(int(classes_[danger_idx]))
            else:
                preds.append(int(classes_[np.argmax(p)]))

        feature_df["predicted_label"] = pd.Series(
            preds,
            index=feature_df.index
        ).astype(int)

        label_map = {
            0: "NORMAL",
            1: "WARNING",
            2: "CRITICAL"
        }

        # 모델이 예측한 라벨을 상태로 매핑
        feature_df["status"] = feature_df["predicted_label"].map(label_map)

        feature_df["received_at"] = feature_df["received_at"].astype(str)

        feature_df = feature_df.rename(columns={
            "bsi": "current_bsi"
        })

        # 출력할 컬럼 선택
        result_cols = [
            "vehicle_id",
            "model_name",
            "received_at",
            "battery_voltage",
            "battery_current",
            "temperature",
            "ambient_temp",
            "delta_i",
            "delta_v",
            "joule_heating_stress",
            "latitude",
            "longitude",
            "current_region_id",
            "region_name",
            "is_active",
            "alert_type",
            "current_bsi",
            "status"
        ]

        result_df = feature_df[result_cols].copy()

        # ----------------------------------------------------
        # current_bsi 타입 정리
        # - SQL decimal/float 저장 안정성 확보
        # ----------------------------------------------------
        result_df["current_bsi"] = pd.to_numeric(
            result_df["current_bsi"],
            errors="coerce"
        ).fillna(0).round(4)

        logging.info(f"=== OUTPUT DF SHAPE: {result_df.shape}")
        logging.info(f"=== OUTPUT PAYLOAD: {str(result_df.to_dict(orient='records'))[:1500]}")

        
        return result_df.to_dict(orient="records")

    except Exception as e:
        logging.error(f"=== 4dt1team_score.py ERROR: {str(e)}", exc_info=True)

        error_row = {
            "vehicle_id": None,
            "model_name": None,
            "received_at": None,
            "battery_voltage": None,
            "battery_current": None,
            "temperature": None,
            "ambient_temp": None,
            "delta_i": None,
            "delta_v": None,
            "joule_heating_stress": None,
            "latitude": None,
            "longitude": None,
            "current_region_id": None,
            "region_name": None,
            "is_active": None,
            "alert_type": None,
            "current_bsi": None,
            "status": "ERROR"
        }
        return [error_row]
