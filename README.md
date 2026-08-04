![banner_rs](./assets/banner_rs.jpg)

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **System Memory:** 256GB RAM
- **Computing Power:** 24-Core / 48-Thread Multi-core CPU
- **GPU:** NVIDIA GeForce RTX 3090 (24GB)
- **NVIDIA Driver Version:** 535.86.10
- **CUDA Version:** 12.2 (Runtime: 11.8)
- **Tool:** VS Code (SSH), Google Colab
- **Language:** Python 3.10.13

### Requirements
```
catboost==1.2.10                                  pandas==2.1.4
fastparquet==2026.3.0                             pyarrow==24.0.0
implicit==0.7.2                                   ray==2.6.3
kmeans_pytorch==0.3                               recbole==1.2.1
lightgbm==4.6.0                                   tqdm==4.66.1
numpy==1.26.2                                     xgboost==3.2.0
```

---

## **📋 Competition Info**
### 커머스 상품 구매 예측 (Commerce Purchase Behavior Prediction)
- 사용자의 쇼핑 패턴을 분석해서 미래(next one week)에 구매할만한 상품 추천
- 이커머스 분야에서 추천 시스템은 사용자의 취향을 분석하여 알맞은 상품을 추천함으로써 사용자의 경험을 증진하고 기업의 매출 향상에 도움을 줄 수 있음

### 일정 (Timeline): 개인 출전 허용
- 2026.05.04 09:00 ~ 2026.05.14 18:00 (Competition)
- 2026.05.15 17:00 ~ 2026.05.15 18:30 (Seminar)

### 데이터셋 정보 (Dataset Info)
- eCommerce behavior data from multi category store 데이터를 전처리하여 사용
- 학습데이터: 2019년 11월 1일부터 2020년 2월 29일까지 4개월간 데이터 (8,350,311건)
- 평가데이터: 2020년 3월 1일부터 2020년 3월 7일까지 일주일간 데이터 (6,382,570건)
- 평가데이터는 cold-start scenario 고려하지 않음: 편의상 train set에 있는 user_id, item_id만 남기고 제거

### 평가지표 (Evaluation Metric)
- NDCG@10: Relevance는 ground-truth set에 따라 1(실제 구매) 0은(구매 안함)으로 나눠짐 (binary relevance)<br>
  즉, 평가데이터의 해당 user가 실제로 구매했으면 1, 아니면 0
- NDCG@10값이 클수록 또는 NDCG@10값이 동일하다면 제출횟수가 적을수록, 높은 순위로 인정
- 평가데이터는 무작위(50:50 random split)로 public testset과 private testset으로 나뉨
- 최종 순위는 private LB를 기반으로 함
- 평가 예시: Most popular Top 10
> 학습데이터에서 가장 많이 구매된 item을 추천<br>
> 모든 user_id에 동일한 10개의 item으로 구성 (비개인화 추천)

### 규정 (Rule)
- 외부 데이터셋 사용 금지
- 사전 학습된 가중치 가져오는 행위 금지
