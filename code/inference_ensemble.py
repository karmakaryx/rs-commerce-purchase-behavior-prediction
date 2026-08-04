import argparse
import time
from collections import defaultdict
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

def log(msg: str, t0: float = None):
    elapsed = f"  (+{time.time() - t0:.1f}s)" if t0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)

def build_features(df: pd.DataFrame, train_hist: pd.DataFrame) -> pd.DataFrame:
    # --- 아이템 인기도 ---
    item_cnt = train_hist.groupby("item_id").size().rename("item_cnt").reset_index()
    df = df.merge(item_cnt, on="item_id", how="left")
    df["item_cnt"] = df["item_cnt"].fillna(0)

    # --- 유저-아이템 전체 interaction 수 ---
    ui_cnt = (train_hist.groupby(["user_id", "item_id"]).size().rename("ui_cnt").reset_index())
    df = df.merge(ui_cnt, on=["user_id", "item_id"], how="left")
    df["ui_cnt"] = df["ui_cnt"].fillna(0)

    # --- event_type 기반 feature ---
    if "event_type" in train_hist.columns:
        cart_hist = train_hist[train_hist["event_type"] == "cart"]
        view_hist = train_hist[train_hist["event_type"] == "view"]

        ui_cart = (cart_hist.groupby(["user_id", "item_id"]).size().rename("ui_cart_cnt").reset_index())
        df = df.merge(ui_cart, on=["user_id", "item_id"], how="left")
        df["ui_cart_cnt"] = df["ui_cart_cnt"].fillna(0)

        ui_view = (view_hist.groupby(["user_id", "item_id"]).size().rename("ui_view_cnt").reset_index())
        df = df.merge(ui_view, on=["user_id", "item_id"], how="left")
        df["ui_view_cnt"] = df["ui_view_cnt"].fillna(0)

        item_cart_cnt = (cart_hist.groupby("item_id").size().rename("item_cart_cnt").reset_index())
        df = df.merge(item_cart_cnt, on="item_id", how="left")
        df["item_cart_cnt"] = df["item_cart_cnt"].fillna(0)
    else:
        df["ui_cart_cnt"] = 0
        df["ui_view_cnt"] = 0
        df["item_cart_cnt"] = 0

    # --- 가격 기반 feature ---
    if "price" in train_hist.columns:
        item_price = (train_hist.groupby("item_id")["price"].mean().rename("item_price").reset_index())
        df = df.merge(item_price, on="item_id", how="left")

        if "event_type" in train_hist.columns:
            purchase_hist = train_hist[train_hist["event_type"] == "purchase"]
        else:
            purchase_hist = train_hist
        user_avg_price = (purchase_hist.groupby("user_id")["price"].mean().rename("user_avg_price").reset_index())
        df = df.merge(user_avg_price, on="user_id", how="left")

        df["price_ratio"] = (
            df["item_price"] / df["user_avg_price"].replace(0, np.nan)
        ).fillna(1.0).clip(0, 10)
        df["item_price"] = df["item_price"].fillna(0)
        df["user_avg_price"] = df["user_avg_price"].fillna(0)
    else:
        df["item_price"] = 0
        df["user_avg_price"] = 0
        df["price_ratio"] = 1.0

    # --- gap_hours: 유저 마지막 활동 vs 아이템 마지막 등장 시간 차 ---
    user_last = (train_hist.groupby("user_id")["event_time"].max().rename("user_last").reset_index())
    item_last = (train_hist.groupby("item_id")["event_time"].max().rename("item_last").reset_index())
    df = df.merge(user_last, on="user_id", how="left").merge(item_last, on="item_id", how="left")
    df["gap_hours"] = (
        (df["user_last"] - df["item_last"]).dt.total_seconds() / 3600.0
    ).fillna(9999).clip(-9999, 9999)

    return df

# train_rerank_lgbm.py와 반드시 동일하게 유지
FEATURES = [
    "sasrec_score", "sasrec_rank",
    "item_cnt", "item_cart_cnt",
    "ui_cnt", "ui_cart_cnt", "ui_view_cnt",
    "item_price", "user_avg_price", "price_ratio",
    "gap_hours",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="../output/candidates.parquet")
    parser.add_argument("--lgbm_model", type=str, default="../output/rerank_lgbm.txt")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--submission_template", type=str, default="../data/sample_submission.csv")
    parser.add_argument("--out", type=str, default="../output/submission_ensemble.csv")
    parser.add_argument("--w_lgbm", type=float, default=0.7)
    parser.add_argument("--w_sasrec", type=float, default=0.3)
    args = parser.parse_args()

    t0 = time.time()

    # 1. candidates 로드
    log("▶ candidates.parquet 로딩 중...", t0)
    df = pd.read_parquet(args.candidates).copy()
    log(f"  candidates shape: {df.shape}  "
        f"(users: {df['user_id'].nunique():,}, "
        f"items/user: ~{len(df) // max(df['user_id'].nunique(), 1)})", t0)

    # 2. feature 재계산
    log("▶ train 데이터 로딩 & feature 생성 중...", t0)
    data_dir = Path(args.data_dir)
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    log(f"  train: {len(train):,}행", t0)

    df = build_features(df, train_hist=train)

    # 누락된 feature 있으면 0으로 보완 (안전장치)
    for c in FEATURES:
        if c not in df.columns:
            log(f"  WARNING: feature '{c}' not found in candidates, filling with 0", t0)
            df[c] = 0

    log(f"  feature 생성 완료: {FEATURES}", t0)

    # 3. LightGBM 모델 로드 & 추론
    log("▶ LightGBM 모델 로딩 중...", t0)
    booster = lgb.Booster(model_file=args.lgbm_model)

    log("▶ LightGBM 점수 예측 중...", t0)
    df["lgbm_score"] = booster.predict(df[FEATURES].fillna(0))
    log(f"  lgbm_score range: [{df['lgbm_score'].min():.4f}, {df['lgbm_score'].max():.4f}]", t0)

    # 4. SASRec rank 정규화 (유저별 0~1 스케일)
    log("▶ SASRec rank 정규화 중...", t0)
    df["sasrec_rank_norm"] = df.groupby("user_id")["sasrec_rank"].rank(
        method="first", ascending=True
    )
    maxr = df.groupby("user_id")["sasrec_rank_norm"].transform("max").replace(0, 1)
    df["sasrec_rank_score"] = 1.0 - (df["sasrec_rank_norm"] - 1) / maxr

    # 5. 앙상블 점수 계산
    log(f"▶ 앙상블 점수 계산 중... (w_lgbm={args.w_lgbm}, w_sasrec={args.w_sasrec})", t0)
    df["final_score"] = args.w_lgbm * df["lgbm_score"] + args.w_sasrec * df["sasrec_rank_score"]

    # 6. 유저별 top-10 추출
    log("▶ 유저별 top-10 추출 중...", t0)
    top10 = (
        df.sort_values(["user_id", "final_score"], ascending=[True, False])
          .groupby("user_id")
          .head(10)[["user_id", "item_id"]]
    )
    log(f"  top10 shape: {top10.shape}", t0)

    # 7. fallback용 인기 아이템 (후보에 없는 유저 대비)
    popular_items = (
        df.groupby("item_id")["sasrec_score"]
          .mean()
          .sort_values(ascending=False)
          .head(10)
          .index.tolist()
    )
    log(f"  fallback popular items: {popular_items[:5]} ...", t0)

    # 8. submission 파일 채우기
    log("▶ submission 파일 생성 중...", t0)
    user_items: dict = defaultdict(list)
    for u, i in top10[["user_id", "item_id"]].itertuples(index=False):
        user_items[u].append(i)

    sub = pd.read_csv(args.submission_template)
    log(f"  submission template shape: {sub.shape}", t0)

    used = defaultdict(int)
    out_items = []
    fallback_count = 0

    for u in sub["user_id"].values:
        arr = user_items.get(u, None)
        if not arr:
            arr = popular_items
            fallback_count += 1
        idx = used[u] % len(arr)
        out_items.append(arr[idx])
        used[u] += 1

    if fallback_count > 0:
        log(f"  WARNING: fallback 적용된 row 수: {fallback_count:,} (popular items로 대체)", t0)

    sub["item_id"] = out_items

    # 9. 저장
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    log(f"▶ 완료! saved: {args.out}", t0)
    log(f"  총 소요시간: {time.time() - t0:.1f}s", t0)

if __name__ == "__main__":
    main()
