import argparse
import json
import os
import pandas as pd
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))

    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    train = train.sort_values(by=["user_id", "event_time"])

    train_df = train[["user_id", "item_id", "user_session", "event_time"]].copy()

    # nanoseconds를 seconds로 변환 (recbole time-based split 정확성 확보)
    train_df["event_time"] = train_df["event_time"].astype("int64") // 10**9

    # sorted() 적용: unique() 첫 등장 순서 의존 제거, 재현성 보장
    user2idx = {u: i for i, u in enumerate(sorted(train_df["user_id"].unique()))}
    item2idx = {it: i for i, it in enumerate(sorted(train_df["item_id"].unique()))}

    with open(os.path.join(args.data_dir, "user2idx.json"), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(args.data_dir, "item2idx.json"), "w") as f:
        json.dump(item2idx, f)

    train_df["user_idx"] = train_df["user_id"].map(user2idx)
    train_df["item_idx"] = train_df["item_id"].map(item2idx)

    train_df = train_df.dropna().reset_index(drop=True)
    train_df.rename(columns={
        "user_idx": "user_idx:token",
        "item_idx": "item_idx:token",
        "event_time": "event_time:float",
    }, inplace=True)

    outdir = os.path.join(args.data_dir, "SASRec_dataset")
    os.makedirs(outdir, exist_ok=True)
    train_df[["user_idx:token", "item_idx:token", "event_time:float"]].to_csv(
        os.path.join(outdir, "SASRec_dataset.inter"),
        sep="\t",
        index=False,
    )
    print("Recbole dataset generated")

if __name__ == "__main__":
    main()
