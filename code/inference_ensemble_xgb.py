import argparse
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd
import xgboost as xgb
from train_rerank_xgb import build_features, FEATURES

def log(msg, t0=None):
    elapsed = f"  (+{time.time() - t0:.1f}s)" if t0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="../output/candidates.parquet")
    parser.add_argument("--xgb_model", type=str, default="../output/rerank_xgb.json")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--submission_template", type=str, default="../data/sample_submission.csv")
    parser.add_argument("--out", type=str, default="../output/submission_ensemble.csv")
    parser.add_argument("--w_xgb", type=float, default=0.7)
    parser.add_argument("--w_sasrec", type=float, default=0.3)
    args = parser.parse_args()

    t0 = time.time()

    # 1. candidates 로드
    log("▶ candidates.parquet 로딩 중...", t0)
    df = pd.read_parquet(args.candidates).copy()
    log(f"  candidates shape: {df.shape}  "
        f"(users: {df['user_id'].nunique():,}, "
        f"items/user: ~{len(df) // max(df['user_id'].nunique(), 1)})", t0)

    # src 컬럼이 없는 이전 버전 candidates 대응
    for col in ["src_cart", "src_repeat", "src_recent", "src_sasrec", "src_popular"]:
        if col not in df.columns:
            log(f"  [WARNING] '{col}' not found in candidates, filling with 0", t0)
            df[col] = 0

    # 2. feature 생성 (전체 train 사용)
    log("▶ train 데이터 로딩 & feature 생성 중...", t0)
    data_dir = Path(args.data_dir)
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    log(f"  train: {len(train):,}행", t0)

    df = build_features(df, train_hist=train)

    for c in FEATURES:
        if c not in df.columns:
            log(f"  [WARNING] feature '{c}' not found in candidates, filling with 0", t0)
            df[c] = 0

    log(f"  feature 생성 완료: {len(FEATURES)})", t0)

    # 3. XGBoost 추론
    log("▶ XGBoost 모델 로딩 중...", t0)
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)

    log("▶ XGBoost 점수 예측 중...", t0)
    dtest = xgb.DMatrix(df[FEATURES].fillna(0).values, feature_names=FEATURES)
    df["xgb_score"] = booster.predict(dtest)
    log(f"  xgb_score range: [{df['xgb_score'].min():.4f}, {df['xgb_score'].max():.4f}]", t0)

    # 4. SASRec rank 정규화 (유저별 0~1 스케일)
    log("▶ SASRec rank 정규화 중...", t0)
    df["sasrec_rank_norm"] = df.groupby("user_id")["sasrec_rank"].rank(
        method="first", ascending=True
    )
    maxr = df.groupby("user_id")["sasrec_rank_norm"].transform("max").replace(0, 1)
    df["sasrec_rank_score"] = 1.0 - (df["sasrec_rank_norm"] - 1) / maxr

    # 5. 앙상블 점수 계산
    log(f"▶ 앙상블 점수 계산 중... (w_xgb={args.w_xgb}, w_sasrec={args.w_sasrec})", t0)
    df["final_score"] = args.w_xgb * df["xgb_score"] + args.w_sasrec * df["sasrec_rank_score"]

    # 6. 유저별 top-10 추출
    log("▶ 유저별 top-10 추출 중...", t0)
    top10 = (
        df.sort_values(["user_id", "final_score"], ascending=[True, False])
          .groupby("user_id")
          .head(10)[["user_id", "item_id"]]
    )
    log(f"  top10 shape: {top10.shape}", t0)

    # 7. fallback 인기 아이템
    popular_items = (
        df.groupby("item_id")["sasrec_score"].mean()
          .sort_values(ascending=False)
          .head(10).index.tolist()
    )
    log(f"  fallback popular items: {popular_items[:5]} ...", t0)

    # 8. submission 채우기
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
        log(f"  [WARNING] fallback 적용된 row 수: {fallback_count:,}", t0)

    sub["item_id"] = out_items

    # 9. 저장
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    log(f"▶ 완료! saved: {args.out}", t0)
    log(f"  총 소요시간: {time.time() - t0:.1f}s", t0)

if __name__ == "__main__":
    main()
