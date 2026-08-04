import argparse
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def log(msg, t0=None):
    elapsed = f"  (+{time.time() - t0:.1f}s)" if t0 else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)

# 메모리 최적화 함수 (Dtype 다운캐스팅)
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    log(f"  메모리 최적화 완료: {start_mem:.1f}MB -> {end_mem:.1f}MB (감소율: {100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df

# Negative Sampling 함수
def apply_negative_sampling(df, n_neg_per_user=50):
    log(f"▶ Negative Sampling 시작 (유저당 label=0 최대 {n_neg_per_user}개 유지)...")

    # label=1(positive)은 무조건 유지
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    # label=0(negative)만 유저별 샘플링
    sampled_neg = neg.groupby("user_id").apply(
        lambda x: x.sample(n=min(len(x), n_neg_per_user), random_state=42)
    ).reset_index(drop=True)

    # 결합 후 순서 섞기
    balanced_df = pd.concat([pos, sampled_neg], ignore_index=True).sample(frac=1, random_state=42)

    log(f"  Sampling 완료: {len(df):,}행 -> {len(balanced_df):,}행")
    return balanced_df

# Feature Engineering (기존 코드와 동일)
RECENT_HOURS = 40.0

def build_features(df: pd.DataFrame, train_hist: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(train_hist["event_time"]):
        train_hist = train_hist.copy()
        train_hist["event_time"] = pd.to_datetime(train_hist["event_time"], utc=True, errors="coerce")

    t_max = train_hist["event_time"].max()
    cutoff_40h = t_max - pd.Timedelta(hours=RECENT_HOURS)

    df = df.copy()
    df["src_priority"] = (df["src_cart"]*5 + df["src_repeat"]*4 + df["src_recent"]*3 + df["src_sasrec"]*2 + df["src_popular"]*1)

    view_hist = train_hist[train_hist["event_type"] == "view"] if "event_type" in train_hist.columns else train_hist
    ui_view = view_hist.groupby(["user_id", "item_id"]).size().rename("ui_view_cnt").reset_index()
    df = df.merge(ui_view, on=["user_id", "item_id"], how="left").fillna({"ui_view_cnt": 0})

    view_40h = view_hist[view_hist["event_time"] >= cutoff_40h]
    ui_view_40h = view_40h.groupby(["user_id", "item_id"]).size().rename("ui_view_cnt_40h").reset_index()
    df = df.merge(ui_view_40h, on=["user_id", "item_id"], how="left").fillna({"ui_view_cnt_40h": 0})

    df["repeat2"] = (df["ui_view_cnt"] >= 2).astype(np.int8)
    df["repeat3"] = (df["ui_view_cnt"] >= 3).astype(np.int8)
    df["repeat5"] = (df["ui_view_cnt"] >= 5).astype(np.int8)

    if "event_type" in train_hist.columns:
        cart_hist = train_hist[train_hist["event_type"] == "cart"]
        ui_cart = cart_hist.groupby(["user_id", "item_id"]).size().rename("_cart_cnt").reset_index()
        df = df.merge(ui_cart, on=["user_id", "item_id"], how="left")
        df["ui_cart_flag"] = (df["_cart_cnt"].fillna(0) > 0).astype(np.int8)
        df.drop(columns=["_cart_cnt"], inplace=True)
    else:
        df["ui_cart_flag"] = 0

    ui_last = view_hist.groupby(["user_id", "item_id"])["event_time"].max().rename("ui_last_time").reset_index()
    df = df.merge(ui_last, on=["user_id", "item_id"], how="left")
    df["ui_last_hours_ago"] = ((t_max - df["ui_last_time"]).dt.total_seconds() / 3600.0).fillna(9999).clip(0, 9999)
    df["ui_last_dow"]  = df["ui_last_time"].dt.dayofweek.fillna(-1).astype(np.int8)
    df["ui_last_hour"] = df["ui_last_time"].dt.hour.fillna(-1).astype(np.int8)
    df.drop(columns=["ui_last_time"], inplace=True)

    item_view_pop = view_hist.groupby("item_id").size().rename("item_view_pop").reset_index()
    df = df.merge(item_view_pop, on="item_id", how="left").fillna({"item_view_pop": 0})

    return df

FEATURES = [
    "sasrec_score", "sasrec_rank", "src_cart", "src_repeat", "src_recent",
    "src_sasrec", "src_popular", "src_priority", "ui_view_cnt", "ui_view_cnt_40h",
    "ui_cart_flag", "repeat2", "repeat3", "repeat5", "ui_last_hours_ago",
    "ui_last_dow", "ui_last_hour", "item_view_pop",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--candidates", type=str, default="../output/candidates.parquet")
    parser.add_argument("--model_out", type=str, default="../output/rerank_xgb.json")
    parser.add_argument("--n_neg_per_user", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)

    # 1. 데이터 로드
    log("▶ 데이터 로딩 중...", t0)
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    cand = pd.read_parquet(args.candidates)

    # 2. 라벨 생성 및 히스토리 분리
    log("▶ 라벨 생성 중...", t0)
    train_sorted = train.sort_values(["user_id", "event_time"])
    last_idx = train_sorted.groupby("user_id").tail(1).index
    train_hist = train_sorted.drop(index=last_idx).copy()
    del train_sorted; gc.collect()

    pos_labels = train[train["event_type"] == "purchase"][["user_id", "item_id"]].drop_duplicates()
    pos_labels["label"] = 1
    df = cand.merge(pos_labels, on=["user_id", "item_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    del train, cand, pos_labels; gc.collect()

    # 3. Negative Sampling
    df = apply_negative_sampling(df, n_neg_per_user=args.n_neg_per_user)
    gc.collect()

    # 4. feature 생성 및 Dtype 최적화
    log("▶ feature 생성 중...", t0)
    df = build_features(df, train_hist)
    df = reduce_mem_usage(df)
    del train_hist; gc.collect()

    # 5. 분리 및 DMatrix 변환 (원본 즉시 삭제)
    log("▶ Train/Valid 분리 및 DMatrix 변환...", t0)
    users = df["user_id"].unique()
    tr_users, va_users = train_test_split(users, test_size=0.2, random_state=42)

    tr = df[df["user_id"].isin(tr_users)].sort_values("user_id")
    va = df[df["user_id"].isin(va_users)].sort_values("user_id")
    del df; gc.collect()

    dtrain = xgb.DMatrix(tr[FEATURES], label=tr["label"])
    dtrain.set_group(tr.groupby("user_id").size().values)
    del tr; gc.collect()

    dval = xgb.DMatrix(va[FEATURES], label=va["label"])
    dval.set_group(va.groupby("user_id").size().values)

    # 6. XGBoost 학습
    log("▶ XGBoost 학습 시작...", t0)
    params = {
        "objective": "rank:ndcg", "eval_metric": "ndcg@10", "eta": 0.05,
        "max_depth": 8, "tree_method": "hist", "device": "cuda",
    }
    booster = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=100)

    # 7. 저장
    booster.save_model(args.model_out)
    log(f"▶ 완료! saved: {args.model_out} (총 소요: {time.time()-t0:.1f}s)", t0)

if __name__ == "__main__":
    main()
