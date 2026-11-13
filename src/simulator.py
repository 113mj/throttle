"""과거 로그 시간순 재생 시뮬레이터.

- 기간: sim 구간 중 하루가 온전한 21~29일차 (30일차는 21시까지만 있음)
- 15분 구간(하루 96개)마다 캠페인별로
    컨트롤러가 노출 여부 결정 -> 남은 예산 안에서 비용 차감 -> 컨트롤러 업데이트
- 예산: 캠페인별 sim 기간 실제 일 평균 비용 x BUDGET_RATIO
- 목표 소진 곡선: train 기간 캠페인별 시간대(15분) 비용 비중의 누적
- 노출 점수: pCVR / 비용 (비용 1당 기대 전환. "pCVR x 목표CPA >= 비용" 입찰 규칙과 같다)
  score="p" 로 바꾸면 pCVR 만 본다. 비싼 노출일수록 pCVR 이 높아서 무작위보다 CPA 가 나쁘다
- 기준 임계값: val 기간(18~20일차)에 보정 없는 일반 MLP 로, 실제 비용의 BUDGET_RATIO
  만큼을 보여주게 되는 점수 값. 확률 단위로 고정해 두고 모델만 바꿔 끼운다.
- 전환: 보여준 노출에 연결된 conversion_id 를 캠페인별로 중복 없이 센다 (기본)
  비교용으로 노출별 라벨 합(중복 포함), 마지막 노출만 인정(last-touch)도 함께 센다
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers import PID, Controller, FixedThreshold, Greedy, Pacing, Step

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DAY = 86400
INTERVAL = 900
N_INT = DAY // INTERVAL           # 96
SIM_DAYS = list(range(21, 30))
BUDGET_RATIO = 0.5
EARLY_K = 88                      # 22시 전에 예산 99% 소진 -> 조기 소진
REF_PRED = "mlp"                  # 기준 임계값을 정하는 모델


@dataclass
class Market:
    campaigns: list
    data: dict          # campaign -> {day: dict(k_bounds, p, cost, conv_id)}
    budget: dict        # campaign -> 일 예산
    target: dict        # campaign -> 길이 96 누적 목표 소진 비율
    tau: dict           # campaign -> 기준 점수 임계값


def score_of(p, cost, score: str):
    return p / cost if score == "p_per_cost" else p


def build_market(pred_col: str, score: str = "p_per_cost", preds_file: str = "preds.parquet") -> Market:
    df = pd.read_parquet(ROOT / "data/sample.parquet",
                         columns=["timestamp", "day", "campaign", "cost", "conversion",
                                  "conversion_id", "conversion_timestamp", "split"])
    preds = pd.read_parquet(ROOT / "artifacts" / preds_file).set_index("row_id")
    df["k"] = (df.timestamp % DAY) // INTERVAL
    conv = df[df.conversion_id != -1]
    df["last_touch"] = 0
    df.loc[conv.groupby("conversion_id").timestamp.idxmax(), "last_touch"] = 1

    # 목표 곡선: train 기간 시간대별 비용 비중
    train = df[df.split == "train"]
    share = train.groupby(["campaign", "k"]).cost.sum().unstack(fill_value=0.0)
    share = share.reindex(columns=range(N_INT), fill_value=0.0)
    target = share.div(share.sum(axis=1), axis=0).cumsum(axis=1)

    # 기준 임계값: val 기간 비용의 BUDGET_RATIO 를 보여주는 점수
    val = df.loc[preds.index[preds.part == "val"]]
    val = val.assign(s=score_of(preds.loc[preds.part == "val", REF_PRED].to_numpy(), val.cost.to_numpy(), score))
    tau = {}
    for c, g in val.groupby("campaign"):
        g = g.sort_values("s", ascending=False)
        cum = g.cost.cumsum() / g.cost.sum()
        tau[c] = float(g.s.iloc[np.searchsorted(cum.to_numpy(), BUDGET_RATIO)])

    sim = df[df.day.isin(SIM_DAYS)].sort_values("timestamp", kind="stable")
    p = preds.loc[sim.index, pred_col].to_numpy()
    sim = sim.assign(p=p, s=score_of(p, sim.cost.to_numpy(), score))
    budget = (sim.groupby("campaign").cost.sum() / len(SIM_DAYS) * BUDGET_RATIO).to_dict()

    data = {}
    for (c, d), g in sim.groupby(["campaign", "day"]):
        data.setdefault(c, {})[d] = {
            "k_bounds": np.searchsorted(g.k.to_numpy(), np.arange(N_INT + 1)),
            "score": g.s.to_numpy(), "cost": g.cost.to_numpy(), "conv_id": g.conversion_id.to_numpy(),
            "label": g.conversion.to_numpy(), "last": g.last_touch.to_numpy(),
            "p": g.p.to_numpy(), "ts": g.timestamp.to_numpy(),
            "conv_ts": g.conversion_timestamp.to_numpy(),
        }
    campaigns = sorted(data)
    return Market(campaigns, data, budget, {c: target.loc[c].to_numpy() for c in campaigns}, tau)


def simulate(market: Market, make: Callable[[object, Market], Controller]) -> dict:
    """반환: log (캠페인 x 일 x 구간 기록), conversions (캠페인별 전환 수), daily (캠페인 x 일)."""
    rows, conversions, daily, credit = [], {}, [], {}
    for c in market.campaigns:
        ctrl = make(c, market)
        budget, target = market.budget[c], market.target[c]
        inc = np.diff(np.concatenate([[0.0], target]))
        conv_ids, label_sum, last_sum = [], 0, 0
        for d in SIM_DAYS:
            day = market.data[c][d]
            b = day["k_bounds"]
            remaining = budget
            n_before = len(conv_ids)
            ctrl.start_day()
            for k in range(N_INT):
                sl = slice(b[k], b[k + 1])
                score, cost = day["score"][sl], day["cost"][sl]
                tau = ctrl.threshold
                want = ctrl.decide(score)
                shown = want & (np.cumsum(np.where(want, cost, 0.0)) <= remaining)
                spend = float(cost[shown].sum())
                remaining -= spend
                conv_ids.append(day["conv_id"][sl][shown])
                label_sum += int(day["label"][sl][shown].sum())
                last_sum += int(day["last"][sl][shown].sum())
                ctrl.observe({k: day[k][sl][shown] for k in ("ts", "p", "label", "conv_ts")})
                spent_frac = 1.0 - remaining / budget
                rows.append((c, d, k, tau, int(shown.sum()), len(score), spend, spent_frac, target[k]))
                ctrl.update(Step(k, spent_frac, target[k], inc[k + 1] if k + 1 < N_INT else 0.0,
                                 remaining, score, cost))
            ctrl.end_day((d + 1) * DAY)
            day_ids = np.concatenate(conv_ids[n_before:])
            daily.append((c, d, int(np.unique(day_ids[day_ids != -1]).size)))
        ids = np.concatenate(conv_ids)
        conversions[c] = int(np.unique(ids[ids != -1]).size)
        credit[c] = (label_sum, last_sum, int((ids != -1).sum()))
    log = pd.DataFrame(rows, columns=["campaign", "day", "k", "threshold", "shown", "impressions",
                                      "spend", "spent_frac", "target_frac"])
    # 날짜별 전환은 그날 안에서만 중복 제거 (날짜를 넘는 중복은 conversions 쪽에서 제거됨)
    daily = pd.DataFrame(daily, columns=["campaign", "day", "conversions"])
    daily = daily.merge(log.groupby(["campaign", "day"], as_index=False).spend.sum())
    return {"log": log, "conversions": conversions, "daily": daily, "credit": credit}


def summarize(result: dict, market: Market) -> dict:
    log = result["log"]
    spend = log.spend.sum()
    budget = sum(market.budget.values()) * len(SIM_DAYS)
    conv = sum(result["conversions"].values())
    cd = log.groupby(["campaign", "day"])
    early = cd.apply(lambda g: bool((g.spent_frac >= 0.99).to_numpy()[:EARLY_K].any()), include_groups=False)
    label_sum, last_sum, conv_impr = np.array(list(result["credit"].values())).sum(axis=0)
    return {
        "conversions": conv,
        "spend": spend,
        "cpa_x1000": 1000 * spend / max(conv, 1),
        "budget_used": spend / budget,
        "shown_share": log.shown.sum() / log.impressions.sum(),
        "pacing_mae_pp": 100 * (log.spent_frac - log.target_frac).abs().mean(),
        "early_depletion": float(early.mean()),
        "conv_label_sum": int(label_sum),        # 노출별 라벨 합 (같은 전환을 여러 번 셈)
        "conv_last_touch": int(last_sum),        # 전환 직전 마지막 노출만 인정
        "impr_per_conversion": conv_impr / max(conv, 1),  # 얻은 전환 1개당 보여준 전환 노출 수
    }


CONTROLLERS = {
    "greedy": lambda c, m: Greedy(),
    "fixed": lambda c, m: FixedThreshold(m.tau[c]),
    "pacing": lambda c, m: Pacing(m.tau[c], seed=hash(c) % 2**32),
    "pid": lambda c, m: PID(m.tau[c]),
}


def plot_curves(results: dict, path: Path, title: str) -> None:
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    hours = (np.arange(N_INT) + 1) / 4
    any_log = next(iter(results.values()))["log"]
    ax.plot(hours, any_log.groupby("k").target_frac.mean(), "--", color=ink2, lw=1.5, label="target")
    for (name, r), col in zip(results.items(), colors):
        ax.plot(hours, r["log"].groupby("k").spent_frac.mean(), color=col, lw=2, label=name)
    ax.set(xlim=(0, 24), ylim=(0, 1.05), xticks=range(0, 25, 3))
    ax.set_xlabel("hour of day", color=ink2)
    ax.set_ylabel("cumulative spend / daily budget", color=ink2)
    ax.set_title(title, color=ink, loc="left")
    ax.grid(color=grid, lw=0.8)
    ax.tick_params(colors=ink2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=ink2, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """동작 확인용: 일반 MLP 로 컨트롤러 4종, 점수 기준 2종 (pCVR / 비용, pCVR 만)."""
    tables = []
    for score in ("p_per_cost", "p"):
        market = build_market(REF_PRED, score)
        results = {name: simulate(market, make) for name, make in CONTROLLERS.items()}
        t = pd.DataFrame({name: summarize(r, market) for name, r in results.items()}).T
        tables.append(t.assign(score=score).rename_axis("controller").reset_index())
        if score == "p_per_cost":
            plot_curves(results, REPORTS / "figures/sim_preview_spend_curve.png",
                        "Average cumulative spend (MLP, score = pCVR / cost, day 21-29)")
    print(f"campaigns={len(market.campaigns)} days={SIM_DAYS[0]}-{SIM_DAYS[-1]} "
          f"total daily budget={sum(market.budget.values()):.2f}")
    table = pd.concat(tables, ignore_index=True)
    print(table.round(4).to_string(index=False))
    table.to_csv(REPORTS / "sim_preview.csv", index=False)
    print("saved reports/sim_preview.csv, reports/figures/sim_preview_spend_curve.png")


if __name__ == "__main__":
    main()
