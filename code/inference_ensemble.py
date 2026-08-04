import argparse
import gc
import time
from pathlib import Path
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoost
from train_rerank_catboost import FEATURES as CB_FEATURES
from train_rerank_xgb import build_features as xgb_build_features, FEATURES as XGB_FEATURES
from train_rerank_lgbm import FEATURES as LGBM_FEATURES
from inference_ensemble_lgbm import build_features as lgbm_build_features

def log(msg, t0=None):
    elapsed = f"  (+{time.time() - t0:.1f}s)" if t0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="../output/candidates_xgb.parquet")
    parser.add_argument("--xgb_model", type=str, default="../output/rerank_xgb.json")
    parser.add_argument("--lgbm_model", type=str, default="../output/rerank_lgbm.txt")
    parser.add_argument("--catboost_model", type=str, default="../output/catboost_rerank.cbm")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--submission_template", type=str, default="../data/sample_submission.csv")
    parser.add_argument("--out", type=str, default="../output/submission_final.csv")
    parser.add_argument("--w_xgb", type=float, default=0.3)
    parser.add_argument("--w_lgbm", type=float, default=0.35)
    parser.add_argument("--w_catboost", type=float, default=0.35)
    parser.add_argument("--w_sasrec", type=float, default=0.0)
    args = parser.parse_args()

    t0 = time.time()

    log("▶ 모델 로딩...", t0)
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(args.xgb_model)
    lgbm_booster = lgb.Booster(model_file=args.lgbm_model)
    cb_model = CatBoost()
    cb_model.load_model(args.catboost_model)

    log("▶ 데이터 로딩...", t0)
    df = pd.read_parquet(args.candidates)

    # 필수 소스 컬럼 누락 방지 및 기본값 채우기
    src_cols = ["src_cart", "src_repeat", "src_recent", "src_sasrec", "src_popular"]
    for col in src_cols:
        if col not in df.columns:
            df[col] = 0

    train = pd.read_parquet(Path(args.data_dir) / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")

    # feature 생성 시 필요한 최소 컬럼 셋 정의
    base_cols = ["user_id", "item_id", "sasrec_score", "sasrec_rank"] + src_cols

    log("▶ XGBoost 추론...", t0)
    X_xgb = xgb_build_features(df[base_cols].copy(), train_hist=train)
    for c in XGB_FEATURES:
        if c not in X_xgb.columns: X_xgb[c] = 0

    dtest = xgb.DMatrix(X_xgb[XGB_FEATURES].fillna(0).values, feature_names=XGB_FEATURES)
    df["xgb_score"] = xgb_booster.predict(dtest).astype("float32")
    del X_xgb, dtest; gc.collect()

    log("▶ LGBM & CatBoost feature 생성...", t0)
    X_shared = lgbm_build_features(df[base_cols].copy(), train_hist=train)
    for c in list(set(LGBM_FEATURES + CB_FEATURES)):
        if c not in X_shared.columns: X_shared[c] = 0

    log("▶ LGBM 추론...", t0)
    df["lgbm_score"] = lgbm_booster.predict(X_shared[LGBM_FEATURES].fillna(0)).astype("float32")

    log("▶ CatBoost 추론...", t0)
    df["catboost_score"] = cb_model.predict(X_shared[CB_FEATURES].fillna(0)).astype("float32")

    del X_shared; gc.collect()

    log("▶ 점수 정규화 및 앙상블...", t0)
    for col in ["xgb_score", "lgbm_score", "catboost_score"]:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn + 1e-8)

    df["final_score"] = (
        args.w_xgb * df["xgb_score"] +
        args.w_lgbm * df["lgbm_score"] +
        args.w_catboost * df["catboost_score"]
    )

    log("▶ top-10 추출...", t0)
    top10 = (
        df.sort_values(["user_id", "final_score"], ascending=[True, False])
          .groupby("user_id")
          .head(10)[["user_id", "item_id"]]
    )
    del df; gc.collect()

    log("▶ submission 생성...", t0)
    user_items = top10.groupby("user_id")["item_id"].apply(list).to_dict()
    popular_items = train["item_id"].value_counts().head(10).index.tolist()

    sub = pd.read_csv(args.submission_template)
    out_items = []
    current_user = None
    item_idx = 0

    for u in sub["user_id"].values:
        items = user_items.get(u, popular_items)
        if u != current_user:
            current_user = u
            item_idx = 0
        out_items.append(items[item_idx] if item_idx < len(items) else popular_items[item_idx % 10])
        item_idx += 1

    sub["item_id"] = out_items
    sub.to_csv(args.out, index=False)
    log(f"▶ 완료: {args.out} ({time.time()-t0:.1f}s)", t0)

if __name__ == "__main__":
    main()
