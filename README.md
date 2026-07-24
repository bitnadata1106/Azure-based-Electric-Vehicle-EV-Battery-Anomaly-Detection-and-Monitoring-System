# Azure-based EV Battery Anomaly Detection & Monitoring System

> 전기차 배터리 주행 데이터를 활용하여 이상 징후를 탐지하고,
> Azure 기반 실시간 데이터 파이프라인과 모니터링 대시보드를 구축한 프로젝트입니다.

<br>

## 1. Project Overview

전기차 배터리는 전압, 전류, 온도, 충전 상태와 같은 여러 지표가 복합적으로 변화하기 때문에 단일 변수만으로 이상 상태를 판단하기 어렵습니다.

본 프로젝트에서는 전기차 배터리 데이터를 분석하여 이상 징후를 나타낼 수 있는 주요 Feature를 선정하고, 이를 기반으로 배터리 상태 지수인 **BSI(Battery Status Index)**를 설계했습니다.

또한 머신러닝 기반 배터리 이상 탐지 모델을 구축하고 Azure Machine Learning Endpoint로 배포하여, 실시간 데이터가 입력되면 배터리 상태를 예측할 수 있는 End-to-End 모니터링 시스템을 구현했습니다.

<br>

## 2. Project Goals

* 배터리 이상 징후와 관련된 주요 Feature 탐색 및 선정
* Z-score 기반 BSI(Battery Status Index) 설계
* 정상·주의·위험 상태를 분류하는 이상 탐지 모델 개발
* Azure Machine Learning을 활용한 모델 배포
* Azure IoT Hub부터 Power BI까지 연결되는 실시간 모니터링 파이프라인 구현

<br>

## 3. Architecture

이미지

<br>

## 4. My Role

본 프로젝트에서 저는 다음 두 가지 업무를 중심으로 수행했습니다.

### 4.1 BSI Feature Selection

배터리 이상 상태를 수치화하기 위한 BSI 공식에 사용할 Feature를 선정했습니다.

단순히 원본 변수를 사용하는 것이 아니라, 배터리 상태 변화와 이상 징후를 더 잘 표현할 수 있도록 파생변수와 시계열 통계값을 생성하고 비교했습니다.

#### 주요 작업

* 배터리 전압, 전류, 온도 데이터 분포 분석
* 정상 구간과 이상 구간의 Feature 변화 비교
* 전압 및 전류 변화량 파생변수 생성
* 전력 및 온도 편차 관련 파생변수 생성
* 이동 평균을 활용한 시계열 변화 추적
* Z-score 기반 Feature 표준화
* BSI 구성에 사용할 주요 변수 선정

#### 주요 Feature

| Feature         | Description                |
| --------------- | -------------------------- |
| `delta_current` | 이전 시점 대비 전류 변화량            |
| `delta_voltage` | 이전 시점 대비 전압 변화량            |
| `abs_power`     | 전압과 전류를 이용한 절대 전력          |
| `temp_diff`     | 배터리 온도 간 편차                |
| `rolling_mean`  | 일정 구간의 이동 평균               |
| `z_score`       | 변수별 평균과 표준편차를 기준으로 계산한 이상도 |

#### BSI Concept

각 Feature의 Z-score에 가중치를 적용하여 하나의 배터리 상태 지수로 통합했습니다.
가중치는 담당 팀원들이 Kaggle NASA 전기배터리 실험 데이터로 추출하였습니다. 

```text
BSI
= w1 × Z(delta_current)
+ w2 × Z(delta_voltage)
+ w3 × Z(abs_power)
+ w4 × Z(temp_diff)
+ w5 × Z(rolling_feature)
```

BSI 점수와 주요 Feature의 변화 패턴을 활용하여 배터리 상태를 다음과 같이 구분했습니다.
BSI를 Z-정규화하여 2sigma 3sigma~ 규칙기반 배터리 상태를 ~ 

```text
Normal  | 정상 범위 |  Z(BSI) < 2sigma
Warning | 이상 징후가 관찰되는 상태 |  2sigma <= Z(BSI) < 3sigma 
Danger  | 즉각적인 확인이 필요한 위험 상태 |  Z(BSI) >= 3sigma
```

<br>

### 4.2 Battery Anomaly Detection Modeling

배터리 상태를 `Normal`, `Warning`, `Danger`로 분류하는 다중 분류 모델을 개발했습니다.
위에서 수립한 배터리 상태를 psedo-label로 하여 ~ 

위험 상태 데이터가 정상 데이터보다 적은 불균형 데이터라는 점을 고려하여 전체 정확도보다 클래스별 Recall과 Macro F1-score를 중심으로 모델을 평가했습니다.

#### 주요 작업

* BMW 전기차 주행 데이터 전처리
* 모델 입력 Feature 구성
* 학습 데이터와 평가 데이터 분리
* LightGBM 기반 다중 분류 모델 학습
* 클래스 불균형을 고려한 성능 평가
* Confusion Matrix를 활용한 오분류 분석
* Danger 클래스 탐지 성능 개선

#### Model

```text
Algorithm : LightGBM Classifier
Task      : Multi-class Classification
Classes   : Normal / Warning / Danger
```

#### Evaluation Metrics

```text
Macro F1-score : 0.743
Danger Recall  : 0.853
```

위험 상태를 정상 또는 주의 상태로 잘못 판단하는 경우 실제 운영 환경에서 안전 문제로 이어질 수 있기 때문에, 본 프로젝트에서는 **Danger Recall을 핵심 평가 지표로 설정했습니다.**

<br>

### 4.3 Model Deployment

학습한 모델을 Azure Machine Learning 환경에 등록하고 실시간 추론 Endpoint로 배포했습니다.

Azure Stream Analytics에서 전달된 배터리 데이터를 모델 Endpoint가 받아 상태를 예측하고, 예측 결과를 Azure SQL Database에 저장하도록 구성했습니다.

#### Deployment Flow

```text
Input Battery Data
        │
        ▼
Feature Schema Validation
        │
        ▼
Azure ML Online Endpoint
        │
        ▼
Anomaly Prediction
        │
        ▼
Azure SQL Database
```

#### 주요 작업

* 학습 모델 Azure ML 등록
* 실시간 추론용 Scoring Script 작성
* 모델 입력 Schema 구성
* Managed Online Endpoint 배포
* REST API 기반 추론 테스트
* Stream Analytics와 Azure ML Endpoint 연결
* 예측 결과 Azure SQL Database 적재

#### Troubleshooting

모델 배포 과정에서 입력 데이터의 배열 구조와 Endpoint가 요구하는 객체 구조가 일치하지 않아 다음과 같은 오류가 발생했습니다.

```text
400 Bad Request
401 Unauthorized
424 Model Error
Input Schema Mismatch
```

Scoring Script의 입력 구조와 Azure ML Endpoint의 요청 Schema를 일치시키고, 추론 함수의 입력 검증 방식을 수정하여 문제를 해결했습니다.

이를 통해 로컬 환경에서 학습한 모델을 클라우드 환경에 배포하고, 실제 데이터 파이프라인과 연결하는 전체 과정을 경험했습니다.

<br>

## 5. Tech Stack

### Language & Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat\&logo=pandas\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat\&logo=scikitlearn\&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)

### Azure

![Azure](https://img.shields.io/badge/Microsoft_Azure-0078D4?style=flat\&logo=microsoftazure\&logoColor=white)
![Azure IoT Hub](https://img.shields.io/badge/Azure_IoT_Hub-0078D4?style=flat\&logo=microsoftazure\&logoColor=white)
![Azure Stream Analytics](https://img.shields.io/badge/Stream_Analytics-0078D4?style=flat\&logo=microsoftazure\&logoColor=white)
![Azure Machine Learning](https://img.shields.io/badge/Azure_Machine_Learning-0078D4?style=flat\&logo=microsoftazure\&logoColor=white)
![Azure SQL Database](https://img.shields.io/badge/Azure_SQL_Database-0078D4?style=flat\&logo=microsoftazure\&logoColor=white)

### Visualization & Collaboration

![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat\&logo=powerbi\&logoColor=black)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat\&logo=github\&logoColor=white)

<br>

## 6. Repository Structure

```text
Azure-based-Electric-Vehicle-EV-Battery-Anomaly-Detection-and-Monitoring-System
│
├── assets
│   ├── architecture.png
│   ├── dashboard.png
│   └── model_result.png
│
├── data_sample
│   └── battery_sample.csv
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_bsi_feature_analysis.ipynb
│   └── 04_lightgbm_modeling.ipynb
│
├── deployment
│   ├── score.py
│   ├── endpoint_test.py
│   └── sample_request.json
│
├── docs
│   ├── bsi_definition.md
│   ├── model_evaluation.md
│   └── troubleshooting.md
│
├── requirements.txt
└── README.md
```

<br>

## 7. Key Takeaways

이번 프로젝트를 통해 단순히 머신러닝 모델을 학습하는 것에서 끝나지 않고 다음과 같은 전체 과정을 경험했습니다.

* 도메인 가설을 기반으로 배터리 이상 Feature 선정
* 시계열 데이터의 이동 평균과 변화량 분석
* Z-score 기반 배터리 상태 지수 설계
* 클래스 불균형을 고려한 모델 평가
* Azure Machine Learning 기반 모델 배포
* 실시간 데이터 파이프라인과 모델 Endpoint 연동
* 배포 과정에서 발생한 Schema 및 API 오류 해결

특히 분석 결과를 모델로 구현하고, 해당 모델을 Azure 환경에 배포하여 실제 서비스 구조에 연결했다는 점에서 의미가 있었습니다.

<br>

## 8. Team Project

본 프로젝트는 Microsoft Data School 과정에서 진행한 팀 프로젝트입니다.

본 저장소는 팀 전체 결과물 중 제가 담당한 **BSI Feature 분석, 배터리 이상 탐지 모델링 및 Azure ML 모델 배포 작업**을 중심으로 재구성한 개인 포트폴리오 저장소입니다.
