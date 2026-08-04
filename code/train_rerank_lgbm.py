import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split

def log(msg: str, t0: float = None):
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
        ideal = np.sort(rel)[::-1]
        idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        out.append(dcg / idcg if idcg > 0 else 0.0)
    n_pos_users = len(out)
    score = float(np.mean(out)) if out else 0.0
    return score, n_pos_users

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--candidates", type=str, default="../output/candidates.parquet")
    parser.add_argument("--model_out", type=str, default="../output/rerank_lgbm.txt")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)

    # 1. 데이터 로드
    log("▶ 데이터 로딩 중...", t0)
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    cand = pd.read_parquet(args.candidates)
    log(f"  train: {len(train):,}행 / candidates: {len(cand):,}행  (users: {cand['user_id'].nunique():,})", t0)

    if "event_type" in train.columns:
        log(f"  event_type 분포:\n{train['event_type'].value_counts().to_string()}", t0)

    # 2. 누수 방지: train을 hist(과거) / last(마지막 1개)로 분리
    log("▶ train/hist 분리 중...", t0)
    train_sorted = train.sort_values(["user_id", "event_time"])
    last_idx = train_sorted.groupby("user_id").tail(1).index
    train_hist = train_sorted.drop(index=last_idx).copy()
    last_rows = train_sorted.loc[last_idx].copy()

    if "event_type" in last_rows.columns:
        pos = (last_rows[last_rows["event_type"] == "purchase"]
               [["user_id", "item_id"]].drop_duplicates())
    else:
        pos = last_rows[["user_id", "item_id"]].drop_duplicates()
    pos = pos.copy()
    pos["label"] = 1

    df = cand.merge(pos, on=["user_id", "item_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    log(f"  positive 샘플: {df['label'].sum():,} / 전체: {len(df):,}", t0)

    # 3. feature 생성 (모두 train_hist 기준)
    log("▶ feature 생성 중...", t0)

    # --- 아이템 인기도 (전체 interaction 수) ---
    item_cnt = (train_hist.groupby("item_id").size().rename("item_cnt").reset_index())
    df = df.merge(item_cnt, on="item_id", how="left")
    df["item_cnt"] = df["item_cnt"].fillna(0)

    # --- 유저-아이템 전체 interaction 횟수 ---
    ui_cnt = (train_hist.groupby(["user_id", "item_id"]).size().rename("ui_cnt").reset_index())
    df = df.merge(ui_cnt, on=["user_id", "item_id"], how="left")
    df["ui_cnt"] = df["ui_cnt"].fillna(0)

    # --- 유저-아이템 cart 횟수 (구매 의도 핵심 신호) ---
    if "event_type" in train_hist.columns:
        cart_hist = train_hist[train_hist["event_type"] == "cart"]
        ui_cart = (cart_hist.groupby(["user_id", "item_id"]).size().rename("ui_cart_cnt").reset_index())
        df = df.merge(ui_cart, on=["user_id", "item_id"], how="left")
        df["ui_cart_cnt"] = df["ui_cart_cnt"].fillna(0)

        # --- 유저-아이템 view 횟수 ---
        view_hist = train_hist[train_hist["event_type"] == "view"]
        ui_view = (view_hist.groupby(["user_id", "item_id"]).size().rename("ui_view_cnt").reset_index())
        df = df.merge(ui_view, on=["user_id", "item_id"], how="left")
        df["ui_view_cnt"] = df["ui_view_cnt"].fillna(0)

        # --- 아이템 cart 인기도 (아이템 단위) ---
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

    features = [
        "sasrec_score", "sasrec_rank",
        "item_cnt", "item_cart_cnt",
        "ui_cnt", "ui_cart_cnt", "ui_view_cnt",
        "item_price", "user_avg_price", "price_ratio",
        "gap_hours",
    ]
    for c in features:
        if c not in df.columns:
            df[c] = 0

    log(f"  feature 목록: {features}", t0)
    log(f"  label=1 비율: {df['label'].mean():.6f}", t0)

    # 4. Train / Validation split (유저 기준)
    users = df["user_id"].drop_duplicates().values
    tr_users, va_users = train_test_split(users, test_size=0.2, random_state=42)

    # sort_values를 split 직후에 수행해야 group 순서와 X/y 순서가 일치함
    tr = df[df["user_id"].isin(tr_users)].sort_values("user_id").copy()
    va = df[df["user_id"].isin(va_users)].sort_values("user_id").copy()

    X_tr, y_tr = tr[features].fillna(0), tr["label"].values
    X_va, y_va = va[features].fillna(0), va["label"].values

    g_tr = tr.groupby("user_id", sort=True).size().values
    g_va = va.groupby("user_id", sort=True).size().values

    log(f"  train users: {len(tr_users):,} / valid users: {len(va_users):,}", t0)

    # 5. LGBMRanker 학습
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    log(f"▶ 학습 시작... (early_stopping_rounds={args.early_stopping_rounds})", t0)
    model.fit(
        X_tr, y_tr,
        group=g_tr,
        eval_set=[(X_va, y_va)],
        eval_group=[g_va],
        eval_at=[10],
        callbacks=[
            early_stopping(stopping_rounds=args.early_stopping_rounds),
            log_evaluation(period=100),
        ],
    )
    log(f"  best iteration: {model.best_iteration_}", t0)

    # 6. 평가
    log("▶ 평가 중...", t0)
    va = va.copy()
    va["pred"] = model.predict(X_va)
    score, n_pos = ndcg_at_k(va[["user_id", "label", "pred"]], k=10)
    log(f"  valid ndcg@10: {score:.6f}  (positive 있는 유저: {n_pos:,}명 기준)", t0)

    # feature 중요도 출력
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    log(f"  feature 중요도:\n{importance.to_string()}", t0)

    # 7. 저장
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(args.model_out)
    log(f"▶ 완료! saved: {args.model_out}  총 소요: {time.time()-t0:.1f}s", t0)

if __name__ == "__main__":
    main()
