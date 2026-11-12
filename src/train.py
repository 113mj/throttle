"""pCVR 모델 학습 + 캘리브레이션 비교.

  fit  : train 구간 중 day 0-17  (모델 학습)
  val  : train 구간 중 day 18-20 (early stopping, 캘리브레이션 보정 학습)
  test : sim 구간 day 21-30      (평가, 시뮬레이터 입력)

모델: 로지스틱 회귀(베이스라인), 범주형 임베딩 + MLP,
      MLP-ds (비전환 10%만 남겨 학습 -> 확률이 부풀려짐, 실험 2 용)
보정: temperature scaling, isotonic regression (val 에서 학습),
      MLP-ds 는 다운샘플링 보정식 p = q / (q + (1-q)/w) 도 비교

출력:
  artifacts/                    모델, 인코더, 보정기, val/sim 예측값
  reports/model_metrics.csv     AUC, LogLoss, ECE, 평균 예측 vs 실제
  reports/figures/reliability.png
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample.parquet"
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"

SEED = 42
VAL_DAYS = 3
CAT_COLS = ["campaign", *[f"cat{i}" for i in range(1, 10)]]
MIN_COUNT = 10        # 이보다 드문 범주는 0번(OOV)으로 묶는다. cat7 은 5.7만 종류
EMB_DIM = 8
HIDDEN = (128, 64)
BATCH = 4096
LR = 1e-3
MAX_EPOCHS = 10
PATIENCE = 2
NEG_RATE = 0.1        # 다운샘플링 모델: 비전환 노출을 이 비율만 남기고 학습
ECE_BINS = 20        # 동일 개수(equal-mass) 구간. 기저 전환율이 6% 라 동일 폭 구간은 앞쪽에 몰린다


# ---------- 피처 ----------

def fit_encoders(df: pd.DataFrame) -> dict:
    vocab = {}
    for c in CAT_COLS:
        vc = df[c].value_counts()
        keep = vc[vc >= MIN_COUNT].index
        vocab[c] = {int(v): i + 1 for i, v in enumerate(keep)}
    tslc = np.log1p(df.time_since_last_click.clip(lower=0))
    return {"vocab": vocab, "tslc_mean": float(tslc.mean()), "tslc_std": float(tslc.std())}


def load_encoders(path: Path = ART / "encoders.json") -> dict:
    """JSON 은 dict 키를 문자열로 저장하므로 vocab 키를 int 로 되돌린다."""
    enc = json.loads(path.read_text())
    enc["vocab"] = {c: {int(k): v for k, v in m.items()} for c, m in enc["vocab"].items()}
    return enc


def encode(df: pd.DataFrame, enc: dict) -> tuple[np.ndarray, np.ndarray]:
    """범주형 인덱스 (n, 10) int64, 수치형 (n, 2) float32."""
    cats = np.stack([
        df[c].map(enc["vocab"][c]).fillna(0).to_numpy(np.int64) for c in CAT_COLS
    ], axis=1)
    t = df.time_since_last_click
    tslc = (np.log1p(t.clip(lower=0)) - enc["tslc_mean"]) / enc["tslc_std"]
    num = np.stack([(t >= 0).astype(np.float32), tslc.astype(np.float32)], axis=1)
    return cats, num


def to_sparse(cats: np.ndarray, num: np.ndarray, sizes: list[int]) -> sp.csr_matrix:
    """로지스틱 회귀용 one-hot + 수치형."""
    offsets = np.cumsum([0, *sizes[:-1]])
    n = len(cats)
    rows = np.repeat(np.arange(n), cats.shape[1])
    onehot = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, (cats + offsets).ravel())),
                           shape=(n, sum(sizes)))
    return sp.hstack([onehot, sp.csr_matrix(num)], format="csr")


# ---------- 모델 ----------

class PCVRNet(nn.Module):
    def __init__(self, sizes: list[int], n_num: int = 2):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(s, EMB_DIM) for s in sizes)
        layers, d = [], EMB_DIM * len(sizes) + n_num
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, cats: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e(cats[:, i]) for i, e in enumerate(self.embs)] + [num], dim=1)
        return self.mlp(x).squeeze(1)  # logit


@torch.no_grad()
def predict_logit(model: nn.Module, cats: np.ndarray, num: np.ndarray) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(cats[i:i + 65536]), torch.from_numpy(num[i:i + 65536]))
           for i in range(0, len(cats), 65536)]
    return torch.cat(out).numpy()


def downsample_negatives(n_rows: int, y: np.ndarray, rate: float) -> np.ndarray:
    """전환(양성)은 전부, 비전환(음성)은 rate 비율만 남기는 행 인덱스."""
    keep = (y == 1) | (np.random.default_rng(SEED).random(n_rows) < rate)
    return np.flatnonzero(keep)


def train_mlp(sizes, fit_x, fit_y, val_x, val_y, logit_offset: float = 0.0, n_num: int = 2) -> PCVRNet:
    """logit_offset: early stopping 판단 시 val 로짓에 더할 값 (다운샘플링 보정용)."""
    model = PCVRNet(sizes, n_num)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    cats, num = map(torch.from_numpy, fit_x)
    y = torch.from_numpy(fit_y.astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(len(y), generator=g).split(BATCH):
            opt.zero_grad()
            loss_fn(model(cats[idx], num[idx]), y[idx]).backward()
            opt.step()
        val_ll = log_loss(val_y, sigmoid(predict_logit(model, *val_x) + logit_offset))
        print(f"  epoch {epoch}: val logloss {val_ll:.5f}")
        if val_ll < best - 1e-5:
            best, bad = val_ll, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


# ---------- 캘리브레이션 / 지표 ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(logit: np.ndarray, y: np.ndarray) -> float:
    nll = lambda log_t: log_loss(y, np.clip(sigmoid(logit / np.exp(log_t)), 1e-7, 1 - 1e-7))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS):
    """동일 개수 구간별 (평균 예측, 실제 전환율, 구간 비중)."""
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    mp = np.array([p[c].mean() for c in chunks])
    my = np.array([y[c].mean() for c in chunks])
    w = np.array([len(c) for c in chunks]) / len(p)
    return mp, my, w


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    mp, my, w = reliability(p, y)
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "ece": float(np.sum(w * np.abs(mp - my))),
        "mean_pred": float(p.mean()),
        "actual_rate": float(y.mean()),
    }


# ---------- 메인 ----------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    df = pd.read_parquet(SAMPLE)
    train_days = df.loc[df.split == "train", "day"]
    val_start = int(train_days.max()) - VAL_DAYS + 1
    part = np.where(df.split == "sim", "test", np.where(df.day < val_start, "fit", "val"))
    parts = {k: df[part == k] for k in ("fit", "val", "test")}
    for k, d in parts.items():
        print(f"{k:4s} days {d.day.min()}-{d.day.max()}  rows={len(d):,}  cvr={d.conversion.mean():.2%}")

    enc = fit_encoders(parts["fit"])
    sizes = [len(enc["vocab"][c]) + 1 for c in CAT_COLS]
    print("vocab sizes:", dict(zip(CAT_COLS, sizes)))
    X = {k: encode(d, enc) for k, d in parts.items()}
    Y = {k: d.conversion.to_numpy() for k, d in parts.items()}

    print("logistic regression ...")
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr.fit(to_sparse(*X["fit"], sizes), Y["fit"])
    lr_logit = {k: lr.decision_function(to_sparse(*X[k], sizes)) for k in ("val", "test")}

    print("MLP ...")
    model = train_mlp(sizes, X["fit"], Y["fit"], X["val"], Y["val"])
    mlp_logit = {k: predict_logit(model, *X[k]) for k in ("val", "test")}

    # 실무에서 흔한 음성 다운샘플링 학습 -> 확률이 부풀려진 모델 (실험 2 용)
    ds = downsample_negatives(len(Y["fit"]), Y["fit"], NEG_RATE)
    ds_offset = float(np.log(NEG_RATE))  # q/(q+(1-q)/w) 보정식 == 로짓 + log(w)
    print(f"MLP-ds (negatives {NEG_RATE:.0%}): {len(ds):,} rows, cvr={Y['fit'][ds].mean():.2%}")
    model_ds = train_mlp(sizes, tuple(x[ds] for x in X["fit"]), Y["fit"][ds],
                         X["val"], Y["val"], logit_offset=ds_offset)
    ds_logit = {k: predict_logit(model_ds, *X[k]) for k in ("val", "test")}

    # 보정기는 val 에서만 학습한다
    t_lr = fit_temperature(lr_logit["val"], Y["val"])
    t_mlp = fit_temperature(mlp_logit["val"], Y["val"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(sigmoid(mlp_logit["val"]), Y["val"])
    iso_ds = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso_ds.fit(sigmoid(ds_logit["val"]), Y["val"])
    print(f"temperature: LR={t_lr:.3f}  MLP={t_mlp:.3f}")

    preds = {
        "LR": {k: sigmoid(v) for k, v in lr_logit.items()},
        "LR + temp": {k: sigmoid(v / t_lr) for k, v in lr_logit.items()},
        "MLP": {k: sigmoid(v) for k, v in mlp_logit.items()},
        "MLP + temp": {k: sigmoid(v / t_mlp) for k, v in mlp_logit.items()},
        "MLP + isotonic": {k: iso.predict(sigmoid(v)) for k, v in mlp_logit.items()},
        "MLP-ds": {k: sigmoid(v) for k, v in ds_logit.items()},
        "MLP-ds + formula": {k: sigmoid(v + ds_offset) for k, v in ds_logit.items()},
        "MLP-ds + isotonic": {k: iso_ds.predict(sigmoid(v)) for k, v in ds_logit.items()},
    }

    rows = [{"model": m, "split": s, **metrics(p[s], Y[s])}
            for m, p in preds.items() for s in ("val", "test")]
    table = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "model_metrics.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table)

    # ---- 저장 ----
    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "sizes": sizes}, ART / "mlp.pt")
    torch.save({"state_dict": model_ds.state_dict(), "sizes": sizes}, ART / "mlp_ds.pt")
    (ART / "encoders.json").write_text(json.dumps(enc))
    (ART / "calibration.json").write_text(json.dumps(
        {"temperature_mlp": t_mlp, "temperature_lr": t_lr, "neg_rate": NEG_RATE, "ds_logit_offset": ds_offset}))
    for name, obj in (("isotonic.pkl", iso), ("isotonic_ds.pkl", iso_ds)):
        with open(ART / name, "wb") as f:
            pickle.dump(obj, f)
    out = []
    for s in ("val", "test"):
        d = pd.DataFrame({"row_id": parts[s].index, "part": s})
        for m, p in preds.items():
            d[m.lower().replace(" + ", "_").replace("-", "_").replace(" ", "_")] = p[s].astype(np.float32)
        out.append(d)
    pd.concat(out).to_parquet(ART / "preds.parquet", index=False)

    plot_reliability(preds, Y["test"], REPORTS / "figures/reliability.png",
                     {"LR": "#2a78d6", "MLP": "#eb6834", "MLP + temp": "#1baf7a", "MLP + isotonic": "#6250d6"},
                     "Reliability on sim period (day 21-30)")
    plot_reliability(preds, Y["test"], REPORTS / "figures/reliability_downsampled.png",
                     {"MLP-ds": "#2a78d6", "MLP-ds + formula": "#eb6834", "MLP-ds + isotonic": "#1baf7a"},
                     f"Negative downsampling ({NEG_RATE:.0%}) model, sim period")


def plot_reliability(preds: dict, y: np.ndarray, path: Path, colors: dict, title: str) -> None:
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    hi = 0
    for name, c in colors.items():
        mp, my, _ = reliability(preds[name]["test"], y)
        ece = metrics(preds[name]["test"], y)["ece"]
        ax.plot(mp, my, "-o", color=c, lw=2, ms=4, label=f"{name}  (ECE {ece:.4f})")
        hi = max(hi, mp.max(), my.max())
    hi *= 1.05
    ax.plot([0, hi], [0, hi], "--", color=ink2, lw=1, label="perfect calibration")
    ax.set(xlim=(0, hi), ylim=(0, hi))
    ax.set_xlabel("mean predicted pCVR (20 equal-mass bins)", color=ink2)
    ax.set_ylabel("actual conversion rate", color=ink2)
    ax.set_title(title, color=ink, loc="left")
    ax.grid(color=grid, lw=0.8)
    ax.tick_params(colors=ink2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=ink2, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
