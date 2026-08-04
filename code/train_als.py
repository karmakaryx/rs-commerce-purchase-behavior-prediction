import argparse
import os
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--num_factor", type=int, default=32, help="The number of latent factors to compute")
    parser.add_argument("--regularization", type=float, default=0.01, help="The regularization factor to use (float)")
    parser.add_argument("--alpha", type=float, default=10.0, help="Governs the baseline confidence in preference observations")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))

    # sorted() 적용: recbole_dataset.py 와 동일하게 재현성 보장
    user2idx = {u: i for i, u in enumerate(sorted(train_df["user_id"].unique()))}
    idx2user = {i: u for u, i in user2idx.items()}
    item2idx = {it: i for i, it in enumerate(sorted(train_df["item_id"].unique()))}
    idx2item = {i: it for it, i in item2idx.items()}

    train_df["user_idx"] = train_df["user_id"].map(user2idx)
    train_df["item_idx"] = train_df["item_id"].map(item2idx)

    train_df["label"] = 1
    user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

    sparse_user_item = sparse.csr_matrix(
        (user_item_matrix["label"].values, (user_item_matrix["user_idx"].values, user_item_matrix["item_idx"].values)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32,
    ).tocsr()

    # GPU 없는 환경에서도 실행 가능하도록 fallback 처리
    try:
        model = AlternatingLeastSquares(
            factors=args.num_factor,
            regularization=args.regularization,
            alpha=args.alpha,
            use_gpu=True,
        )
        model.fit(sparse_user_item)
    except Exception as e:
        print(f"[WARNING] GPU 학습 실패 ({e}), CPU로 재시도합니다.")
        model = AlternatingLeastSquares(
            factors=args.num_factor,
            regularization=args.regularization,
            alpha=args.alpha,
            use_gpu=False,
        )
        model.fit(sparse_user_item)

    test_users_idx = np.array(train_df["user_idx"].unique())
    test_users_idx_li = [num for num in test_users_idx for _ in range(10)]
    public_outputs = model.recommend(
        test_users_idx,
        sparse_user_item[test_users_idx],
        N=10,
        filter_already_liked_items=False,
    )

    recommend_items = public_outputs[0]
    sub_df = pd.DataFrame({
        "user_id": test_users_idx_li,
        "item_id": recommend_items.flatten(),
    })
    sub_df["user_id"] = sub_df["user_id"].map(idx2user)
    sub_df["item_id"] = sub_df["item_id"].map(idx2item)

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    sub_df.to_csv(os.path.join(outdir, "als_output.csv"), index=False)
    print(f"saved: {os.path.join(outdir, 'als_output.csv')}")

if __name__ == "__main__":
    main()
