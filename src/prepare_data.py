"""Criteo Attribution 데이터 샘플링 + 시간순 분할 + 간단한 EDA.

원본: https://huggingface.co/datasets/criteo/criteo-attribution-dataset
(Criteo AI Lab 페이지의 원래 다운로드 링크는 더 이상 동작하지 않음)

출력:
  data/sample.parquet          샘플 (split 컬럼: train / sim)
  reports/eda_campaigns.csv    캠페인별 노출 수, 전환율, 평균 비용
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/criteo_attribution_dataset.tsv.gz"
OUT = ROOT / "data/sample.parquet"
REPORTS = ROOT / "reports"

DAY = 86400
TOP_CAMPAIGNS = 20
UID_KEEP_PCT = 40      # uid 단위로 잘라야 사용자 노출 이력이 끊기지 않는다
UID_SALT = 42          # 샘플링 시드 역할
TRAIN_FRAC = 0.7

# click_pos, click_nb 는 전환 이후에 정해지는 값이라 버린다
COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp",
    "conversion_id", "attribution", "click", "cost", "cpo",
    "time_since_last_click", *[f"cat{i}" for i in range(1, 10)],
]


def uid_bucket(uid: pd.Series) -> np.ndarray:
    """uid -> 0~99 결정적 버킷 (Knuth multiplicative hash)."""
    h = (uid.to_numpy(dtype=np.uint64) + np.uint64(UID_SALT)) * np.uint64(2654435761)
    return ((h >> np.uint64(16)) % np.uint64(100)).astype(np.int64)


def split_day(days: pd.Series, frac: float) -> int:
    """누적 행 비율이 frac 에 가장 가까운 날짜 경계. 이 날짜부터 sim."""
    cum = days.value_counts().sort_index().cumsum() / len(days)
    return int((cum - frac).abs().idxmin()) + 1


def campaign_eda(df: pd.DataFrame) -> pd.DataFrame:
    n_days = df.groupby("split")["day"].nunique()
    g = df.groupby(["campaign", "split"]).agg(
        impressions=("uid", "size"),
        cvr=("conversion", "mean"),
        unique_conversions=("conversion_id", lambda s: s[s != -1].nunique()),
        avg_cost=("cost", "mean"),
        total_cost=("cost", "sum"),
    )
    g["daily_cost"] = g["total_cost"] / g.index.get_level_values("split").map(n_days)
    g = g.unstack("split")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.sort_values("impressions_train", ascending=False)


def main(raw: Path) -> None:
    df = pd.read_csv(raw, sep="\t", engine="pyarrow", usecols=COLUMNS)
    print(f"raw: {len(df):,} rows, {df.campaign.nunique()} campaigns")

    top = df.campaign.value_counts().head(TOP_CAMPAIGNS).index
    df = df[df.campaign.isin(top)]
    df = df[uid_bucket(df.uid) < UID_KEEP_PCT].sort_values("timestamp", kind="stable")
    df = df.reset_index(drop=True)

    df["day"] = (df.timestamp // DAY).astype(np.int16)
    cut = split_day(df.day, TRAIN_FRAC)
    df["split"] = np.where(df.day < cut, "train", "sim")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"sample: {len(df):,} rows, {df.uid.nunique():,} users -> {OUT.relative_to(ROOT)}")
    print(f"split: train day 0-{cut - 1}, sim day {cut}-{df.day.max()}")
    for name, part in df.groupby("split", sort=False):
        conv = part.conversion_id[part.conversion_id != -1].nunique()
        print(f"  {name:5s} rows={len(part):>9,} ({len(part) / len(df):.1%})  "
              f"cvr={part.conversion.mean():.2%}  unique_conv={conv:,}  "
              f"avg_cost={part.cost.mean():.2e}")

    # train 라벨 중 전환이 sim 구간에 일어난 비율 (지연 전환에 의한 누수 크기)
    train = df[df.split == "train"]
    late = (train.conversion_timestamp >= cut * DAY).sum() / max(train.conversion.sum(), 1)
    print(f"train conversions observed after cutoff: {late:.1%}")

    eda = campaign_eda(df)
    REPORTS.mkdir(exist_ok=True)
    eda.to_csv(REPORTS / "eda_campaigns.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4g}".format):
        print(eda[["impressions_train", "impressions_sim", "cvr_train", "cvr_sim",
                   "avg_cost_train", "daily_cost_sim"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=RAW)
    main(p.parse_args().raw)
