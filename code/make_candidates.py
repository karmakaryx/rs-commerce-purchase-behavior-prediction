import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm

def load_maps(data_dir: Path):
    with open(data_dir / "user2idx.json", "r") as f:
        user2idx = json.load(f)
    with open(data_dir / "item2idx.json", "r") as f:
        item2idx = json.load(f)

    idx2user = {int(v): k for k, v in user2idx.items()}
    idx2item = {int(v): k for k, v in item2idx.items()}
    return user2idx, item2idx, idx2user, idx2item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--out", type=str, default="../output/candidates.parquet")
    args = parser.parse_args()

    original_cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (original_cwd / data_dir).resolve()

    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")

    user2idx, item2idx, idx2user, idx2item = load_maps(data_dir)

    train["user_idx"] = train["user_id"].map(user2idx)
    train["item_idx"] = train["item_id"].map(item2idx)
    train = train.dropna(subset=["user_idx", "item_idx"]).copy()
    train["user_idx"] = train["user_idx"].astype(int)
    train["item_idx"] = train["item_idx"].astype(int)

    # 인기도 기준 정렬
    pop = train.groupby("item_idx").size().sort_values(ascending=False)
    # numpy int64를 Python int로 변환하여 dict key mismatch 방지
    pop_items = [int(x) for x in pop.index.tolist()]

    model_file = Path(args.model_file)
    if not model_file.is_absolute():
        model_file = (original_cwd / model_file).resolve()

    os.chdir(script_dir)
    try:
        config, model, dataset, _, _, test_data = load_data_and_model(model_file=str(model_file))
    finally:
        os.chdir(original_cwd)

    print("Data and model load complete")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (original_cwd / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("sasrec_score", pa.float32()),
        ("sasrec_rank", pa.int32()),
    ])
    writer = pq.ParquetWriter(str(out_path), schema)

    batch_rows = []
    batch_size = 5000

    users = sorted(train["user_idx"].unique().tolist())

    for i, uidx in enumerate(tqdm(users)):
        if str(uidx) in dataset.field2token_id["user_idx"]:
            rec_u = dataset.token2id(dataset.uid_field, str(uidx))
            topk_score, topk_iid = full_sort_topk(
                [rec_u], model, test_data, k=args.k, device=config["device"]
            )
            scores = topk_score.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
            token_matrix = dataset.id2token(dataset.iid_field, topk_iid.cpu())
            items = [int(x) for x in token_matrix[-1]]
        else:
            items, scores = [], []

        # 부족하면 인기도 순으로 채움
        s_items = set(items)
        for p in pop_items:
            if len(items) >= args.k:
                break
            if p not in s_items:
                items.append(p)
                scores.append(-999.0)
                s_items.add(p)

        u_id_str = idx2user[int(uidx)]
        for r, (iid, s) in enumerate(zip(items, scores), start=1):
            batch_rows.append({
                "user_id": u_id_str,
                "item_id": idx2item[int(iid)],
                "sasrec_score": s,
                "sasrec_rank": r,
            })

        if (i + 1) % batch_size == 0 and batch_rows:
            df_batch = pd.DataFrame(batch_rows)
            df_batch["sasrec_score"] = df_batch["sasrec_score"].astype("float32")
            df_batch["sasrec_rank"] = df_batch["sasrec_rank"].astype("int32")
            table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
            writer.write_table(table)
            batch_rows = []

    if batch_rows:
        df_batch = pd.DataFrame(batch_rows)
        df_batch["sasrec_score"] = df_batch["sasrec_score"].astype("float32")
        df_batch["sasrec_rank"] = df_batch["sasrec_rank"].astype("int32")
        table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
        writer.write_table(table)

    writer.close()
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
