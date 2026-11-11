"""PyTorch pCVR 모델 -> ONNX 변환, 출력 일치 확인, 배치 크기별 추론 속도 측정.

서빙 그래프에 sigmoid 와 보정(다운샘플링 모델은 로짓 + log(w))까지 넣어서
ONNX 출력이 바로 컨트롤러가 쓰는 확률이 되도록 한다.

출력:
  artifacts/mlp.onnx, artifacts/mlp_ds.onnx
  reports/onnx_benchmark.csv
  reports/figures/onnx_throughput.png
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from train import PCVRNet, encode, load_encoders

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REPORTS = ROOT / "reports"

BATCH_SIZES = [1, 16, 128, 1024, 8192]
BENCH_SECONDS = 2.0
THREADS = 1                # 단일 코어 기준 (보수적인 수치)
TARGET_QPS = 30_000_000 / 86400  # 하루 3,000만 건 ≈ 초당 347건


class Serving(nn.Module):
    """범주형/수치형 입력 -> 보정된 pCVR."""

    def __init__(self, net: PCVRNet, logit_offset: float = 0.0):
        super().__init__()
        self.net = net
        self.register_buffer("offset", torch.tensor(logit_offset))

    def forward(self, cats: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(cats, num) + self.offset)


def load_serving(name: str, offset: float) -> Serving:
    ckpt = torch.load(ART / f"{name}.pt")
    net = PCVRNet(ckpt["sizes"])
    net.load_state_dict(ckpt["state_dict"])
    return Serving(net, offset).eval()


def export(model: Serving, example: tuple, path: Path) -> None:
    batch = {0: "batch"}
    torch.onnx.export(model, example, path, input_names=["cats", "num"], output_names=["pcvr"],
                      dynamic_axes={"cats": batch, "num": batch, "pcvr": batch},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    print(f"exported {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} KB)")


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = THREADS
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench(fn, cats: np.ndarray, num: np.ndarray, bs: int) -> dict:
    """BENCH_SECONDS 동안 bs 크기 배치를 반복 추론. 처리량과 호출당 지연시간."""
    n = len(cats)
    fn(cats[:bs], num[:bs])  # warm-up
    lat, rows, i = [], 0, 0
    t_end = time.perf_counter() + BENCH_SECONDS
    while time.perf_counter() < t_end:
        s = i % (n - bs)
        t0 = time.perf_counter()
        fn(cats[s:s + bs], num[s:s + bs])
        lat.append(time.perf_counter() - t0)
        rows += bs
        i += bs
    total = sum(lat)
    return {"rows_per_sec": rows / total,
            "latency_p50_ms": 1e3 * float(np.percentile(lat, 50)),
            "latency_p99_ms": 1e3 * float(np.percentile(lat, 99))}


def main() -> None:
    torch.set_num_threads(THREADS)
    enc = load_encoders()
    calib = json.loads((ART / "calibration.json").read_text())
    df = pd.read_parquet(ROOT / "data/sample.parquet")
    test = df[df.split == "sim"]
    cats, num = encode(test, enc)

    models = {"mlp": load_serving("mlp", 0.0),
              "mlp_ds": load_serving("mlp_ds", calib["ds_logit_offset"])}
    example = (torch.from_numpy(cats[:2]), torch.from_numpy(num[:2]))

    # ---- 변환 + 출력 일치 확인 (sim 구간 전체) ----
    preds = pd.read_parquet(ART / "preds.parquet").set_index("row_id")
    for name, model in models.items():
        path = ART / f"{name}.onnx"
        export(model, example, path)
        with torch.no_grad():
            p_torch = model(torch.from_numpy(cats), torch.from_numpy(num)).numpy()
        p_onnx = session(path).run(None, {"cats": cats, "num": num})[0]
        ref = preds.loc[test.index, "mlp" if name == "mlp" else "mlp_ds_formula"].to_numpy()
        d_torch, d_ref = np.abs(p_torch - p_onnx).max(), np.abs(ref - p_onnx).max()
        print(f"  {name}: rows={len(p_onnx):,}  max|torch-onnx|={d_torch:.2e}  max|train.py-onnx|={d_ref:.2e}")
        assert d_torch < 1e-5 and d_ref < 1e-5, f"{name}: ONNX 출력이 학습 시 예측과 다르다"

    # ---- 속도 측정 ----
    sess = session(ART / "mlp.onnx")
    model = models["mlp"]

    def run_onnx(c, x):
        return sess.run(None, {"cats": c, "num": x})

    @torch.no_grad()
    def run_torch(c, x):
        return model(torch.from_numpy(c), torch.from_numpy(x))

    rows = []
    for bs in BATCH_SIZES:
        for engine, fn in (("PyTorch", run_torch), ("ONNX Runtime", run_onnx)):
            r = {"engine": engine, "batch_size": bs, **bench(fn, cats, num, bs)}
            r["x_target"] = r["rows_per_sec"] / TARGET_QPS
            rows.append(r)
            print(f"  bs={bs:>5}  {engine:12s}  {r['rows_per_sec']:>12,.0f} rows/s  "
                  f"p50={r['latency_p50_ms']:.3f}ms  p99={r['latency_p99_ms']:.3f}ms  ({r['x_target']:,.0f}x target)")
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "onnx_benchmark.csv", index=False)
    plot(table, REPORTS / "figures/onnx_throughput.png")


def plot(table: pd.DataFrame, path: Path) -> None:
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"PyTorch": "#2a78d6", "ONNX Runtime": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for engine, c in colors.items():
        t = table[table.engine == engine]
        ax.plot(t.batch_size, t.rows_per_sec, "-o", color=c, lw=2, ms=5)
        last = t.iloc[-1]
        ax.annotate(engine, (last.batch_size, last.rows_per_sec), xytext=(6, 0),
                    textcoords="offset points", va="center", color=ink2)
    ax.axhline(TARGET_QPS, color=ink2, ls="--", lw=1)
    ax.annotate(f"target {TARGET_QPS:,.0f} rows/s (30M/day)", (1, TARGET_QPS), xytext=(0, 5),
                textcoords="offset points", color=ink2)
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("batch size", color=ink2)
    ax.set_ylabel(f"rows / sec ({THREADS} CPU thread)", color=ink2)
    ax.set_title("pCVR inference throughput", color=ink, loc="left")
    ax.grid(color=grid, lw=0.8, which="major")
    ax.tick_params(colors=ink2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
