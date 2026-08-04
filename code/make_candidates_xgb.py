import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from recbole.utils.case_study import full_sort_topk
from tqdm import tqdm

# SASRec 로드 (없어도 동작)
def _try_load_sasrec(model_file, script_dir, original_cwd):
    if model_file is None:
        return None, None, None
    try:
        from recbole.quick_start.quick_start import load_data_and_model
        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = (original_cwd / model_path).resolve()
        os.chdir(script_dir)
        try:
            config, model, dataset, _, _, test_data = load_data_and_model(
                model_file=str(model_path)
            )
        finally:
            os.chdir(original_cwd)
        print("SASRec loaded:", model_path)
        return config, model, dataset, test_data
    except Exception as e:
        print(f"[WARNING] SASRec 로드 실패 ({e}), SASRec 없이 진행합니다.")
        return None, None, None, None

def _sasrec_scores(uid_list, config, model, dataset, test_data, k):
    result = {}
    for rec_u in uid_list:
        topk_score, topk_iid = full_sort_topk(
            [rec_u], model, test_data, k=k, device=config["device"]
        )
        scores = topk_score.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
        token_matrix = dataset.id2token(dataset.iid_field, topk_iid.cpu())
        if token_matrix.size == 0:
            result = {}
            continue
        token_flat = token_matrix[-1] if token_matrix.ndim > 1 else token_matrix
        items = [int(x) for x in token_flat]
        result = dict(zip(items, scores))
    return result

# 후보 생성 메인 로직
def build_candidates_for_user(
    uidx: int,
    user_events: pd.DataFrame,
    sasrec_score_map: dict,
    pop_items: list,
    k_cart: int,
    k_repeat: int,
    k_recent: int,
    k_sasrec: int,
    k_popular: int,
    recent_hours: float,
) -> list[dict]:
    seen: dict[int, dict] = {}

    def add(item, src_key):
        if item not in seen:
            seen[item] = {
                "src_cart": 0, "src_repeat": 0,
                "src_recent": 0, "src_sasrec": 0, "src_popular": 0,
                "sasrec_score": sasrec_score_map.get(item, -999.0),
                "sasrec_rank": 9999,
            }
        seen[item][src_key] = 1

    # 1. cart
    cart_items = (
        user_events[user_events["event_type"] == "cart"]["item_idx"].value_counts().head(k_cart).index.tolist()
    )
    for it in cart_items:
        add(int(it), "src_cart")

    # 2. repeat view
    view_cnt = (
        user_events[user_events["event_type"] == "view"]["item_idx"].value_counts()
    )
    repeat_items = view_cnt[view_cnt >= 2].head(k_repeat).index.tolist()
    for it in repeat_items:
        add(int(it), "src_repeat")

    # 3. recent view (최근 recent_hours 시간)
    if "event_time" in user_events.columns and len(user_events) > 0:
        t_max = user_events["event_time"].max()
        cutoff = t_max - pd.Timedelta(hours=recent_hours)
        recent_items = (
            user_events[
                (user_events["event_type"] == "view") &
                (user_events["event_time"] >= cutoff)
            ]["item_idx"].value_counts().head(k_recent).index.tolist()
        )
        for it in recent_items:
            add(int(it), "src_recent")

    # 4. SASRec
    sasrec_ranked = sorted(sasrec_score_map.items(), key=lambda x: x[1], reverse=True)
    for rank, (it, _) in enumerate(sasrec_ranked[:k_sasrec], start=1):
        add(int(it), "src_sasrec")
        seen[int(it)]["sasrec_rank"] = rank

    # 5. popular fallback
    added_popular = 0
    for it in pop_items:
        if added_popular >= k_popular:
            break
        add(int(it), "src_popular")
        added_popular += 1

    return [{"item_idx": k, **v} for k, v in seen.items()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_dataset", type=str, default="train.parquet")
    parser.add_argument("--model_file", type=str, default=None, help="SASRec .pth 경로 (없으면 SASRec 없이 진행)")
    parser.add_argument("--k_cart", type=int, default=50)
    parser.add_argument("--k_repeat", type=int, default=200)
    parser.add_argument("--k_recent", type=int, default=200)
    parser.add_argument("--k_sasrec", type=int, default=200)
    parser.add_argument("--k_popular", type=int, default=200)
    parser.add_argument("--recent_hours", type=float, default=40.0, help="recent view 윈도우 (시간 단위, 기본 40h)")
    parser.add_argument("--out", type=str, default="../output/candidates.parquet")
    args = parser.parse_args()

    original_cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (original_cwd / data_dir).resolve()

    print("▶ train 로딩...")
    train = pd.read_parquet(data_dir / args.train_dataset)
    train["event_time"] = pd.to_datetime(train["event_time"], utc=True, errors="coerce")

    with open(data_dir / "user2idx.json") as f:
        user2idx = json.load(f)
    with open(data_dir / "item2idx.json") as f:
        item2idx = json.load(f)

    idx2user = {int(v): k for k, v in user2idx.items()}
    idx2item = {int(v): k for k, v in item2idx.items()}

    train["user_idx"] = train["user_id"].map(user2idx)
    train["item_idx"] = train["item_id"].map(item2idx)
    train = train.dropna(subset=["user_idx", "item_idx"]).copy()
    train["user_idx"] = train["user_idx"].astype(int)
    train["item_idx"] = train["item_idx"].astype(int)

    # 인기도: 구매 횟수 기준, 없으면 전체 interaction
    if "event_type" in train.columns:
        purchase_df = train[train["event_type"] == "purchase"]
        if len(purchase_df) > 0:
            pop_series = purchase_df.groupby("item_idx").size().sort_values(ascending=False)
        else:
            pop_series = train.groupby("item_idx").size().sort_values(ascending=False)
    else:
        pop_series = train.groupby("item_idx").size().sort_values(ascending=False)
    pop_items = [int(x) for x in pop_series.index.tolist()]

    # SASRec 로드
    sasrec_result = _try_load_sasrec(args.model_file, script_dir, original_cwd)
    if sasrec_result[0] is not None:
        config, model, dataset, test_data = sasrec_result
        use_sasrec = True
    else:
        use_sasrec = False

    # 출력 스키마
    schema = pa.schema([
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("sasrec_score", pa.float32()),
        ("sasrec_rank", pa.int32()),
        ("src_cart", pa.int8()),
        ("src_repeat", pa.int8()),
        ("src_recent", pa.int8()),
        ("src_sasrec", pa.int8()),
        ("src_popular", pa.int8()),
    ])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (original_cwd / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema)

    # 유저별 처리
    grouped = {uid: grp for uid, grp in train.groupby("user_idx")}
    users = sorted(grouped.keys())
    batch_rows, BATCH = [], 5000

    for i, uidx in enumerate(tqdm(users)):
        user_events = grouped[uidx]

        # SASRec 점수 맵
        if use_sasrec and str(uidx) in dataset.field2token_id["user_idx"]:
            rec_u = dataset.token2id(dataset.uid_field, str(uidx))
            k_total = args.k_sasrec
            sasrec_map = _sasrec_scores(
                [rec_u], config, model, dataset, test_data, k_total,
            )
        else:
            sasrec_map = {}

        cands = build_candidates_for_user(
            uidx, user_events, sasrec_map, pop_items,
            k_cart = args.k_cart,
            k_repeat = args.k_repeat,
            k_recent = args.k_recent,
            k_sasrec = args.k_sasrec,
            k_popular = args.k_popular,
            recent_hours = args.recent_hours,
        )

        u_id_str = idx2user[uidx]
        for c in cands:
            batch_rows.append({
                "user_id": u_id_str,
                "item_id": idx2item[c["item_idx"]],
                "sasrec_score": float(c["sasrec_score"]),
                "sasrec_rank": int(c["sasrec_rank"]),
                "src_cart": c["src_cart"],
                "src_repeat": c["src_repeat"],
                "src_recent": c["src_recent"],
                "src_sasrec": c["src_sasrec"],
                "src_popular": c["src_popular"],
            })

        if (i + 1) % BATCH == 0 and batch_rows:
            _flush(batch_rows, schema, writer)
            batch_rows = []

    if batch_rows:
        _flush(batch_rows, schema, writer)

    writer.close()
    print(f"saved: {out_path}")

def _flush(rows, schema, writer):
    df = pd.DataFrame(rows)
    df["sasrec_score"] = df["sasrec_score"].astype("float32")
    df["sasrec_rank"] = df["sasrec_rank"].astype("int32")
    for c in ["src_cart", "src_repeat", "src_recent", "src_sasrec", "src_popular"]:
        df[c] = df[c].astype("int8")
    writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))

if __name__ == "__main__":
    main()
