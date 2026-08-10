![banner_rs](./assets/banner_rs.jpg)

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **System Memory:** 256GB RAM
- **Computing Power:** 24-Core / 48-Thread Multi-core CPU
- **GPU:** NVIDIA GeForce RTX 3090 (24GB VRAM)
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
- 사용자의 쇼핑 패턴을 분석해서 각 사용자에게 미래(next one week)에 구매 가능성 높은 상위 10개 아이템 추천
- 이커머스 분야에서 추천 시스템은 사용자의 취향을 분석하여 알맞은 상품을 추천함으로써 사용자의 경험을 증진하고 기업의 매출 향상에 도움을 줄 수 있음

### 일정 (Timeline): 개인 출전 허용
- 2026.05.04 09:00 ~ 2026.05.14 18:00 (Competition)
- 2026.05.15 17:00 ~ 2026.05.15 18:30 (Seminar)

### 데이터셋 정보 (Dataset Info)
- eCommerce behavior data from multi category store 데이터를 전처리하여 사용
- 학습데이터: 2019년 11월 1일부터 2020년 2월 29일까지 4개월간 데이터 (8,350,311건)
- 평가데이터: 2020년 3월 1일부터 2020년 3월 7일까지 일주일간 데이터 (6,382,570건)
- cold-start scenario 고려하지 않음: 편의상 train set에 있는 user_id, item_id만 남기고 제거

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

---

## **💾 Data Description**
### EDA (Exploratory Data Analysis)
#### 1. Property Description
- user_id: Permanent user ID
- item_id: ID of a product
- user_session: Temporary user's session ID. Same for each user's session. Is changed every time user come back to online store from a long pause.
- event_time: Time when event happened at (in UTC)
- category_code: Product's category taxonomy (code name) if it was possible to make it. Usually present for meaningful categories and skipped for different kinds of accessories.
- brand: Down-cased string of brand name. Can be missed.
- price: Float price of a product. Present.
- event_type: Only one kind of event - purchase

#### 2. 사용자 행동 기록
- 사용자는(user_id)는 홈페이지에 들어가 세션(user_session)을 할당받고 특정 아이템(item_id)을 특정 시간(event_time)에 상품(product_id) 장바구니에 추가(event_type='cart')하거나 조회(event_type='view')하거나 구매(event_type='purchase')할 수 있음
- 각 상품별로 (해당 시점에 따른) 카테고리 코드(category_code)와 브랜드(brand), 가격(price)이 주어짐

#### 3. submission.csv: 다음 일주일 동안 사용자가 구매한 상품 예측
- 구매한 (event_type='purchase') 상품의 예측이 목적
- 제출 파일은 user_id, item_id로 구성, user_id 당 top 10개의 item_id를 정렬해서 구성해야 함 (predicted score 기준으로 item_id가 내림차순 정렬되어 있어야 함)
- 학습데이터에 포함된 모든 user에 대해서 예측을 진행하므로, 해당 파일은 총 6,382,570(638,257명의 user에게 10건씩) row로 구성되고 모든 유저에게 10건씩 중복 없이 item을 추천해야만 채점이 진행

#### 4. 기초 통계량
> interactions: 8,350,311<br>
> unique users: 638,257<br>
> unique items: 29,502<br>
> unique sessions: 2,889,552<br>
> category codes: 24<br>
> brands: 1,859

#### 5. 이벤트 타입
- 데이터 희소성 99.96%로 구매는 전체의 0.02% 밖에 안됨
> view: 8,331,873 (99.78%)<br>
> cart: 16,362 (0.20%)<br>
> purchase: 2,076 (0.02%)

#### 6. View2Cart, View2Purchase, Cart2Purchase Rates
- 장바구니에 담으면 12.68% 확률로 구매하므로 cart 이벤트가 강력한 구매 신호
> View2Cart rate: 0.0019<br>
> View2Purchase rate: 0.0002<br>
> Cart2Purchase rate: 0.1268

#### 7. 유저별 상호작용 분포
> 평균: 13.08회 / 중앙값: 6회 / 최대: 37,207회<br>
> 5회 미만 상호작용 유저: 268,279명 (42.03%)<br>
> cold start 문제: 42%의 유저가 필터링됨 (SASRec 학습시 이 유저들이 제외됨)
![eda_user](./assets/eda_user.png)

#### 6. 시간에 따른 이벤트
> 주말(토, 일) 상호작용이 가장 많지만 목요일에 구매 전환율이 특이하게 높음 (0.072%)
![eda_time](./assets/eda_time.png)

#### 7. 카테고리와 브랜드
- 모든 데이터가 의류(apparel) 카테고리이므로 카테고리 정보는 차별화에 도움 안됨
> 메인 카테고리: 1개 (apparel만 존재)<br>
> 서브 카테고리: 24개

#### 8. 조회 인기 브랜드 vs 구매 인기 브랜드
- 구매 수가 너무 적어 브랜드가 큰 의미는 없어 보임 (최대 224개)
> `respect`는 조회 1위지만 구매는 5위: 구경만 많이 함<br>
> `xiaomi`, `sony`, `samsung`이 실제 구매로 이어지는 브랜드<br>
> `iqos`, `glo` 같은 전자담배가 구매 상위에 있음
![eda_brand](./assets/eda_brand.png)

#### 9. 아이템별 상호작용 통계
> long-tail 분포: 상위 10% 아이템이 전체 상호작용의 63.4%를 차지
![eda_item](./assets/eda_item.png)
> 전체 아이템: 29,502개<br>
> 구매된 아이템: 996개 (3.38%)<br>
> 구매안된 아이템: 28,506개 (96.62%)

#### 10. 가격 통계
> 대부분 가격은 500이하로 책정되어 있으며, 구매된 상품의 가격이 더 낮음<br>
> $0-50 가격대가 구매 전환율 가장 높음
![eda_price](./assets/eda_price.png)

#### 11. 문제점 요약
| 문제 | 심각도 | 영향 | 개선 방향 |
| :--- | :--- | :--- | :--- |
| 구매 데이터 부족 (0.02%) | 🔴 심각 | 학습 신호 부족 | 이벤트 가중치 |
| cold start (42%) | 🔴 심각 | 점수 동일 원인 | 필터링 완화 |
| long-tail (63%) | 🟡 중간 | 다양성 부족 | 인기도 조절 |
| 희소성 (99.96%) | 🟡 중간 | 협업필터링 한계 | 하이브리드 모델 |
| 카테고리 무의미 | 🟢 낮음 | 특징 손실 | 무시 가능 |

### Data Preprocessing
