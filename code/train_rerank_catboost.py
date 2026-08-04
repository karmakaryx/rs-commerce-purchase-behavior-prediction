import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from sklearn.model_selection import train_test_split
from inference_ensemble_lgbm import build_features as lgbm_build_features

FEATURES = [
    "sasrec_score", "sasrec_rank",
    "item_cnt", "item_cart_cnt",
    "ui_cnt", "ui_cart_cnt", "ui_view_cnt",
    "item_price", "user_avg_price", "price_ratio",
    "gap_hours",
]

def log(msg, t0=None):
    elapsed = f"  (+{time.time() - t0:.1f}s)" if t0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)

def ndcg_at_k(df, k=10):
    out = []
    for _, g in df.groupby("user_id"):
        g = g.sort_values("pred", ascending=False).head(k)
        rel = g["label"].values
        if rel.sum() == 0:
            continue
        dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
        idcg = (np.sort(rel)[::-1] / np.log2(np.arange(2, len(rel) + 2))).sum()
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(out)) if out else 0.0), len(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--candidates", type=str, default="../output/candidates_xgb.parquet")
    parser.add_argument("--model_out", type=str, default="../output/rerank_catboost.cbm")
    parser.add_argument("--n_estimators", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--device", type=str, default="GPU")
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)

    log("▶ 데이터 로딩 중...", t0)
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    cand = pd.read_parquet(args.candidates)
    log(f"  train: {len(train):,}행 / candidates: {len(cand):,}행", t0)

    log("▶ 라벨 생성 중...", t0)
    train_sorted = train.sort_values(["user_id", "event_time"])
    last_idx = train_sorted.groupby("user_id").tail(1).index
    train_hist = train_sorted.drop(index=last_idx).copy()

    if "event_type" in train.columns:
        pos = (train[train["event_type"] == "purchase"]
               [["user_id", "item_id"]].drop_duplicates())
    else:
        pos = train[["user_id", "item_id"]].drop_duplicates()
    pos = pos.copy()
    pos["label"] = 1

    df = cand.merge(pos, on=["user_id", "item_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    log(f"  positive: {df['label'].sum():,} / 전체: {len(df):,}", t0)

    log("▶ feature 생성 중...", t0)
    df = lgbm_build_features(df, train_hist)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0

    # Train / Validation split
    users = df["user_id"].drop_duplicates().values
    tr_users, va_users = train_test_split(users, test_size=0.2, random_state=42)

    tr = df[df["user_id"].isin(tr_users)].sort_values("user_id").copy()
    va = df[df["user_id"].isin(va_users)].sort_values("user_id").copy()

    X_tr, y_tr = tr[FEATURES].fillna(0), tr["label"].values
    X_va, y_va = va[FEATURES].fillna(0), va["label"].values

    g_tr = tr.groupby("user_id", sort=True).size().values.tolist()
    g_va = va.groupby("user_id", sort=True).size().values.tolist()

    log(f"  train users: {len(tr_users):,} / valid users: {len(va_users):,}", t0)

    # CatBoost Pool
    train_pool = Pool(data=X_tr, label=y_tr, group_id=np.repeat(np.arange(len(g_tr)), g_tr))
    val_pool = Pool(data=X_va, label=y_va, group_id=np.repeat(np.arange(len(g_va)), g_va))

    log(f"▶ CatBoost 학습 시작... (device={args.device})", t0)
    model = CatBoost({
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10;type=Exp",
        "iterations": args.n_estimators,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": 3.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "early_stopping_rounds": args.early_stopping_rounds,
        "task_type": args.device,
        "verbose": 100,
        "random_seed": 42,
    })
    model.fit(train_pool, eval_set=val_pool)
    log(f"  best iteration: {model.best_iteration_}", t0)

    log("▶ 평가 중...", t0)
    va = va.copy()
    va["pred"] = model.predict(X_va.fillna(0))
    score, n_pos = ndcg_at_k(va[["user_id", "label", "pred"]], k=10)
    log(f"  valid ndcg@10: {score:.6f}  (positive 유저: {n_pos:,}명)", t0)

    importance = pd.Series(model.get_feature_importance(data=train_pool), index=FEATURES).sort_values(ascending=False)
    log(f"  feature 중요도:\n{importance.to_string()}", t0)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    log(f"▶ 완료! saved: {args.model_out} (총 소요: {time.time()-t0:.1f}s)", t0)

if __name__ == "__main__":
    main()
