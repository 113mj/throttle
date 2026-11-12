"""사용자 이력 피처 실험: 피처를 하나씩 추가하며 MLP 를 다시 학습한다.

이력 피처는 모두 "그 노출 이전" 정보만으로 만든다 (시간순 정렬 후 사용자별 누적).
샘플은 상위 20개 캠페인만 남긴 데이터라, 이력도 이 20개 캠페인 안에서의 이력이다.

출력:
  artifacts/preds_features.parquet  변형별 val/test 예측 + 비교 기준(oracle, cost_only)
  reports/exp5_model_metrics.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import (CAT_COLS, SEED, fit_encoders, encode, metrics, predict_logit, sigmoid,
                   train_mlp)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DAY = 86400
VAL_DAYS = 3

HISTORY = {
    "n_prev": "이 사용자의 이전 노출 수",
    "n_prev_campaign": "같은 캠페인 이전 노출 수",
    "since_prev": "직전 노출 후 경과 시간",
    "since_first": "첫 노출 후 경과 시간",
    "n_prev_clicks": "이전 클릭 수",
}
VARIANTS = {
    "base": [],
    **{f"+{f}": [f] for f in HISTORY},
    "+history": list(HISTORY),
    "+history+hour": [*HISTORY, "hour"],
    "+history+hour+cost": [*HISTORY, "hour", "cost"],
    "lasttouch": [],
    "+history lasttouch": list(HISTORY),
}
# 라벨을 "전환 직전 마지막 노출만 1" 로 바꾼 변형. 평가 지표(AUC 등)는 비교를 위해 원래 라벨로 계산
LAST_TOUCH = {"lasttouch", "+history lasttouch"}


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """df 는 timestamp 순으로 정렬돼 있어야 한다. 각 행 이전의 정보만 쓴다."""
    assert df.timestamp.is_monotonic_increasing
    u = df.groupby("uid", sort=False)
    prev_ts = u.timestamp.shift(1)
    out = pd.DataFrame(index=df.index)
    out["n_prev"] = u.cumcount()
    out["n_prev_campaign"] = df.groupby(["uid", "campaign"], sort=False).cumcount()
    out["since_prev"] = (df.timestamp - prev_ts).fillna(-1)
    out["since_first"] = df.timestamp - u.timestamp.transform("first")
    out["n_prev_clicks"] = u.click.cumsum() - df.click   # 자기 자신의 클릭은 빼고
    out["hour"] = (df.timestamp % DAY) // 3600
    out["cost"] = df.cost
    return out


def check_no_leak(df: pd.DataFrame, h: pd.DataFrame) -> None:
    """첫 노출의 이력은 비어 있어야 하고, 누적값은 줄어들면 안 된다."""
    first = ~df.uid.duplicated()
    assert (h.n_prev[first] == 0).all() and (h.since_first[first] == 0).all()
    assert (h.since_prev[first] == -1).all() and (h.n_prev_clicks[first] == 0).all()
    assert (h.groupby(df.uid).n_prev_clicks.diff().fillna(0) >= 0).all()


def numeric(h: pd.DataFrame, feats: list, stats: dict | None = None):
    """log1p 후 표준화. since_prev 는 '이전 노출 없음' 플래그를 따로 둔다."""
    cols = {}
    for f in feats:
        if f == "hour":
            continue
        if f == "since_prev":
            cols["has_prev"] = (h[f] >= 0).astype(np.float32)
            cols[f] = np.log1p(h[f].clip(lower=0))
        elif f == "cost":
            cols[f] = np.log(h[f])
        else:
            cols[f] = np.log1p(h[f])
    x = pd.DataFrame(cols, index=h.index)
    if stats is None:
        stats = {c: (x[c].mean(), x[c].std() or 1.0) for c in x if c != "has_prev"}
    for c, (m, s) in stats.items():
        x[c] = (x[c] - m) / s
    return x.to_numpy(np.float32), stats


def main() -> None:
    torch.set_num_threads(4)
    df = pd.read_parquet(ROOT / "data/sample.parquet")
    h = history_features(df)
    check_no_leak(df, h)
    print("이력 피처 누수 검사 통과")

    val_start = int(df.loc[df.split == "train", "day"].max()) - VAL_DAYS + 1
    part = np.where(df.split == "sim", "test", np.where(df.day < val_start, "fit", "val"))
    idx = {k: np.flatnonzero(part == k) for k in ("fit", "val", "test")}
    y = {k: df.conversion.to_numpy()[i] for k, i in idx.items()}
    conv = df[df.conversion_id != -1]
    last = np.zeros(len(df), dtype=np.int64)
    last[df.index.get_indexer(conv.groupby("conversion_id").timestamp.idxmax())] = 1
    y_last = {k: last[i] for k, i in idx.items()}
    enc = fit_encoders(df.iloc[idx["fit"]])
    base = {k: encode(df.iloc[i], enc) for k, i in idx.items()}

    out = {k: pd.DataFrame({"row_id": df.index[i], "part": k}) for k, i in idx.items() if k != "fit"}
    rows = []
    for name, feats in VARIANTS.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        num_fit, stats = numeric(h.iloc[idx["fit"]], feats)
        X = {}
        for k, i in idx.items():
            num, _ = numeric(h.iloc[i], feats, stats)
            cats, num0 = base[k]
            if "hour" in feats:
                cats = np.concatenate([cats, h.hour.to_numpy()[i, None] + 1], axis=1)
            X[k] = (cats, np.concatenate([num0, num], axis=1))
        sizes = [len(enc["vocab"][c]) + 1 for c in CAT_COLS] + ([25] if "hour" in feats else [])
        print(f"\n[{name}] numeric={X['fit'][1].shape[1]} categorical={len(sizes)}")
        yy = y_last if name in LAST_TOUCH else y
        model = train_mlp(sizes, X["fit"], yy["fit"], X["val"], yy["val"], n_num=X["fit"][1].shape[1])
        for k in ("val", "test"):
            p = sigmoid(predict_logit(model, *X[k]))
            out[k][name] = p.astype(np.float32)
            rows.append({"variant": name, "split": k, **metrics(p, y[k])})

    # 비교 기준: 모델이 없는 경우(전환율 상수 -> 점수 = 1/비용), 정답을 아는 경우
    for k, i in idx.items():
        if k == "fit":
            continue
        out[k]["cost_only"] = np.float32(y["fit"].mean())
        out[k]["oracle"] = np.clip(y[k], 1e-3, 1).astype(np.float32)
    preds = pd.concat([out["val"], out["test"]], ignore_index=True)
    preds["mlp"] = preds["base"]          # 시뮬레이터 기준 임계값용 (REF_PRED)
    preds.to_parquet(ART / "preds_features.parquet", index=False)

    table = pd.DataFrame(rows)
    test = table[table.split == "test"].set_index("variant")
    table["auc_gain_vs_base"] = table.auc - table.split.map(table[table.variant == "base"].set_index("split").auc)
    table.to_csv(REPORTS / "exp5_model_metrics.csv", index=False)
    print("\n" + test[["auc", "logloss", "ece", "mean_pred", "actual_rate"]].round(4).to_string())
    print("saved artifacts/preds_features.parquet, reports/exp5_model_metrics.csv")


if __name__ == "__main__":
    main()
