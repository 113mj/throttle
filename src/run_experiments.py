"""실험 일괄 실행. 결과는 reports/exp*.csv, reports/figures/exp*.png

  python src/run_experiments.py          # 전부
  python src/run_experiments.py 1 3      # 골라서

1. 컨트롤러 비교 (+ 같은 지출에서의 비교)
2. 캘리브레이션: 정확한 / 부풀려진 / 보정한 모델을 컨트롤러에 넣었을 때
3. PID 게인 (Kp, Ki, Kd) 에 따른 소진 곡선 안정성
4. 전환율 급락: 실제 드리프트 캠페인 + 인위적 급락(3개 캠페인, 25일차부터 전환 50% 제거)
5. 모델 개선 -> 제어 성과: 이력 피처 모델을 PID 에 넣었을 때 (src/train_features.py 먼저 실행)
"""
import copy
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers import PID, FixedThreshold, Pacing
from simulator import (CONTROLLERS, N_INT, SIM_DAYS, Market, build_market, plot_curves,
                       simulate, summarize)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
SEED = 42

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]


@lru_cache(maxsize=None)
def market(pred: str, preds_file: str = "preds.parquet") -> Market:
    return build_market(pred, preds_file=preds_file)


def run(m: Market, make) -> tuple[dict, dict]:
    r = simulate(m, make)
    return r, summarize(r, m)


def axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("  saved figures/" + name)


def show(df: pd.DataFrame, name: str) -> None:
    df.to_csv(REPORTS / name, index=False)
    print(df.round(4).to_string(index=False))
    print("  saved " + name)


def campaign_day_stats(log: pd.DataFrame) -> pd.DataFrame:
    """캠페인-하루별 소진 편차(%p)와 임계값 변동성(구간 간 |Δ log 임계값| 평균)."""
    g = log.assign(err=(log.spent_frac - log.target_frac).abs(),
                   dlog=np.log(log.threshold.clip(lower=1e-12)).groupby([log.campaign, log.day]).diff().abs())
    return g.groupby(["campaign", "day"]).agg(mae_pp=("err", lambda x: 100 * x.mean()),
                                               volatility=("dlog", "mean")).reset_index()


# ---------------------------------------------------------------- 실험 1

def exp1() -> None:
    print("\n[실험 1] 컨트롤러 비교 (일반 MLP, 점수 = pCVR / 비용)")
    m = market("mlp")
    results, rows = {}, []
    for name in ("greedy", "fixed", "pacing", "pid"):
        r, s = run(m, CONTROLLERS[name])
        cd = campaign_day_stats(r["log"])
        results[name] = r
        rows.append({"controller": name, **s, "pacing_mae_p90_pp": cd.mae_pp.quantile(0.9)})
    table = pd.DataFrame(rows)

    # 같은 지출에서 비교: 고정 임계값을 배율로 바꿔가며 (지출, 전환) 곡선을 만든다
    front = []
    for mult in np.geomspace(0.25, 4, 17):
        _, s = run(m, lambda c, mk, mult=mult: FixedThreshold(mk.tau[c] * mult))
        front.append({"tau_multiplier": mult, **s})
    front = pd.DataFrame(front).sort_values("spend")
    front.to_csv(REPORTS / "exp1_fixed_frontier.csv", index=False)
    table["fixed_conv_at_same_spend"] = np.interp(table.spend, front.spend, front.conversions)
    table["conv_vs_fixed_same_spend"] = table.conversions / table.fixed_conv_at_same_spend - 1
    # 임계값을 낮추면 오전에 예산을 소진해 오히려 전환이 줄어든다 -> 고정의 최댓값과도 비교
    table["conv_vs_best_fixed"] = table.conversions / front.conversions.max() - 1
    print(f"  best fixed: x{front.loc[front.conversions.idxmax(), 'tau_multiplier']:.2f}, "
          f"{front.conversions.max():,} conversions")
    show(table, "exp1_controllers.csv")

    plot_curves(results, FIG / "exp1_spend_curves.png", "Average cumulative spend vs target (day 21-29)")
    print("  saved figures/exp1_spend_curves.png")

    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor=SURFACE)
    ax.plot(front.spend, front.conversions, "-", color=INK2, lw=1.5,
            label="fixed threshold, scaled x0.25 ~ x4")
    for (name, r), col in zip(table.set_index("controller").iterrows(), SERIES):
        ax.plot(r.spend, r.conversions, "o", ms=9, color=col, mec=SURFACE, mew=2, label=name)
    axes(ax, "Conversions vs spend: same-spend comparison", "total spend (9 days)", "conversions")
    ax.legend(frameon=False, labelcolor=INK2, loc="lower right")
    save(fig, "exp1_frontier.png")


# ---------------------------------------------------------------- 실험 2

MODELS = {"mlp": "calibrated MLP", "mlp_ds": "downsampled, no fix (x4.5)",
          "mlp_ds_formula": "downsampled + formula", "mlp_ds_isotonic": "downsampled + isotonic"}


def exp2() -> None:
    print("\n[실험 2] 캘리브레이션: 모델만 바꿔 끼우기 (임계값은 일반 MLP 기준으로 고정)")
    rows, daily, logs = [], [], {}
    for pred in MODELS:
        m = market(pred)
        for name in ("fixed", "pacing", "pid"):
            r, s = run(m, CONTROLLERS[name])
            rows.append({"model": pred, "controller": name, **s})
            d = r["daily"].groupby("day")[["spend", "conversions"]].sum()
            early = r["log"].groupby(["campaign", "day"]).apply(
                lambda g: bool((g.spent_frac >= 0.99).to_numpy()[:88].any()), include_groups=False)
            d["early_depletion"] = early.groupby("day").mean()
            daily.append(d.reset_index().assign(model=pred, controller=name))
            logs[(pred, name)] = r["log"]
    show(pd.DataFrame(rows), "exp2_calibration.csv")
    daily = pd.concat(daily)
    daily["cpa_x1000"] = 1000 * daily.spend / daily.conversions
    daily.to_csv(REPORTS / "exp2_daily.csv", index=False)
    piv = daily[daily.model.isin(["mlp", "mlp_ds"])].pivot_table(
        index="day", columns=["model", "controller"], values="cpa_x1000")
    print("  일별 CPA (x1000):")
    print(piv.round(2).to_string())

    # PID 가 부풀려진 확률을 얼마나 빨리 배우나: 임계값 / 기준 임계값 (캠페인 중앙값)
    tau0 = pd.Series(market("mlp").tau)
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    for (pred, col) in (("mlp", SERIES[0]), ("mlp_ds", SERIES[1])):
        lg = logs[(pred, "pid")]
        ratio = lg.threshold / lg.campaign.map(tau0)
        t = (lg.day - SIM_DAYS[0]) + lg.k / N_INT
        med = ratio.groupby(t).median()
        ax.plot(med.index, med.values, color=col, lw=1.6, label=MODELS[pred])
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8], labels=["x0.5", "x1", "x2", "x4", "x8"])
    ax.minorticks_off()
    ax.axhline(1, color=INK2, ls="--", lw=1)
    axes(ax, "PID threshold / initial threshold (median over campaigns)",
         f"days since start (day {SIM_DAYS[0]} = 0)", "threshold multiplier (log)")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    save(fig, "exp2_pid_threshold.png")


# ---------------------------------------------------------------- 실험 3

def exp3() -> None:
    print("\n[실험 3] PID 게인")
    m = market("mlp")
    rows, traces = [], {}
    for kd in (0.0, 4.0):
        for ki in (0.0, 0.2, 1.0):
            for kp in (0.5, 1, 2, 4, 8, 16, 32):
                r, s = run(m, lambda c, mk, kp=kp, ki=ki, kd=kd: PID(mk.tau[c], kp=kp, ki=ki, kd=kd))
                cd = campaign_day_stats(r["log"])
                rows.append({"kp": kp, "ki": ki, "kd": kd, **s,
                             "pacing_mae_p90_pp": cd.mae_pp.quantile(0.9),
                             "threshold_volatility": cd.volatility.mean()})
                if ki == 0.2 and kd == 0.0 and kp in (0.5, 4, 32):
                    traces[kp] = r["log"]
    table = pd.DataFrame(rows)
    show(table[["kp", "ki", "kd", "pacing_mae_pp", "pacing_mae_p90_pp", "threshold_volatility",
                "cpa_x1000", "budget_used", "early_depletion", "conversions"]], "exp3_pid_gains.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    base = table[table.kd == 0]
    for ki, col in zip((0.0, 0.2, 1.0), SERIES):
        t = base[base.ki == ki]
        a1.plot(t.kp, t.pacing_mae_pp, "-o", color=col, lw=2, ms=4, label=f"Ki={ki}")
        a2.plot(t.kp, t.threshold_volatility, "-o", color=col, lw=2, ms=4, label=f"Ki={ki}")
    for a in (a1, a2):
        a.set_xscale("log")
    axes(a1, "Pacing error (lower = closer to target)", "Kp (log)", "mean |spent - target| (%p)")
    axes(a2, "Threshold volatility (higher = jumpier)", "Kp (log)", "mean |Δ log threshold| per 15 min")
    a1.legend(frameon=False, labelcolor=INK2)
    save(fig, "exp3_gain_tradeoff.png")

    # 예산이 가장 큰 캠페인의 하루 궤적
    c = max(m.budget, key=m.budget.get)
    d = SIM_DAYS[1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor=SURFACE)
    hours = (np.arange(N_INT) + 1) / 4
    first = next(iter(traces.values()))
    one = first[(first.campaign == c) & (first.day == d)]
    a2.plot(hours, one.target_frac, "--", color=INK2, lw=1.2, label="target")
    for (kp, lg), col in zip(traces.items(), SERIES):
        one = lg[(lg.campaign == c) & (lg.day == d)]
        a1.plot(hours, one.threshold / m.tau[c], color=col, lw=1.6, label=f"Kp={kp}")
        a2.plot(hours, one.spent_frac, color=col, lw=1.6, label=f"Kp={kp}")
    a1.set_yscale("log")
    a1.minorticks_off()
    a1.set_yticks([0.5, 0.7, 1, 1.4, 2], labels=["x0.5", "x0.7", "x1", "x1.4", "x2"])
    axes(a1, f"Campaign {c}, day {d}: threshold (Ki=0.2, Kd=0)", "", "threshold / initial (log)")
    axes(a2, "Cumulative spend / budget", "hour of day", "spent fraction")
    a1.legend(frameon=False, labelcolor=INK2)
    a2.legend(frameon=False, labelcolor=INK2, loc="lower right")
    a2.set_xticks(range(0, 25, 3))
    save(fig, "exp3_threshold_traces.png")


# ---------------------------------------------------------------- 실험 4

SHOCK_DAY = 25
SHOCK_DROP = 0.5
DRIFT_CAMPAIGNS = (26852339, 6686701)   # 실제로 sim 구간에서 전환율이 급락한 캠페인


class KnowsShock(FixedThreshold):
    """참고용 '정답을 아는' 컨트롤러: 급락일부터 임계값을 1/(1-drop) 배로 올린다."""
    name = "oracle"

    def __init__(self, tau: float, shocked: bool):
        super().__init__(tau)
        self.tau0, self.shocked, self.day = tau, shocked, SIM_DAYS[0] - 1

    def start_day(self) -> None:
        self.day += 1
        if self.shocked and self.day >= SHOCK_DAY:
            self.threshold = self.tau0 / (1 - SHOCK_DROP)


def shocked_market(base: Market, campaigns: list) -> Market:
    m = copy.deepcopy(base)
    rng = np.random.default_rng(SEED)
    for c in campaigns:
        for d in SIM_DAYS:
            if d < SHOCK_DAY:
                continue
            ids = m.data[c][d]["conv_id"]
            uniq = np.unique(ids[ids != -1])
            dropped = uniq[rng.random(uniq.size) < SHOCK_DROP]
            m.data[c][d]["conv_id"] = np.where(np.isin(ids, dropped), -1, ids)
    return m


def exp4() -> None:
    print("\n[실험 4] 전환율 급락")
    base = market("mlp")

    # (a) 실제 드리프트 캠페인: 모델이 믿는 전환 vs 실제 전환 (보여준 노출 기준)
    df = pd.read_parquet(ROOT / "data/sample.parquet", columns=["day", "campaign", "cost", "conversion"])
    preds = pd.read_parquet(ROOT / "artifacts/preds.parquet").set_index("row_id")
    sim = df[df.day.isin(SIM_DAYS)].join(preds["mlp"])
    per = sim.groupby("campaign").agg(actual_cvr=("conversion", "mean"), pred_cvr=("mlp", "mean"))
    per["pred_over_actual"] = per.pred_cvr / per.actual_cvr
    rows = []
    for name in ("fixed", "pacing", "pid"):
        r, _ = run(base, CONTROLLERS[name])
        spend = r["daily"].groupby("campaign").spend.sum()
        for c in base.campaigns:
            rows.append({"controller": name, "campaign": c, "pred_over_actual": per.loc[c, "pred_over_actual"],
                         "budget_used": spend[c] / (base.budget[c] * len(SIM_DAYS)),
                         "cpa_x1000": 1000 * spend[c] / r["conversions"][c]})
    real = pd.DataFrame(rows).pivot_table(index=["campaign", "pred_over_actual"], columns="controller",
                                          values=["budget_used", "cpa_x1000"])
    real.columns = [f"{v}_{c}" for v, c in real.columns]
    real = real.reset_index().sort_values("pred_over_actual")
    print("  (a) 캠페인별: 모델 과대예측 배율과 CPA")
    show(real, "exp4_real_drift.csv")

    # (b) 인위적 급락: 예산 상위 3개 캠페인, 25일차부터 전환 50% 제거
    shocked = sorted(base.budget, key=base.budget.get, reverse=True)[:3]
    sm = shocked_market(base, shocked)
    makers = dict((k, CONTROLLERS[k]) for k in ("fixed", "pacing", "pid"))
    makers["oracle"] = lambda c, mk: KnowsShock(mk.tau[c], c in shocked)
    rows, daily = [], []
    for name, make in makers.items():
        for label, mk in (("no shock", base), ("shock", sm)):
            r, _ = run(mk, make)
            d = r["daily"][r["daily"].campaign.isin(shocked)]
            d = d.assign(period=np.where(d.day < SHOCK_DAY, "before", "after"))
            g = d.groupby("period")[["spend", "conversions"]].sum()
            for period in ("before", "after"):
                rows.append({"controller": name, "scenario": label, "period": period,
                             "spend_per_day": g.loc[period, "spend"] / d[d.period == period].day.nunique(),
                             "cpa_x1000": 1000 * g.loc[period, "spend"] / g.loc[period, "conversions"]})
            daily.append(d.groupby("day")[["spend", "conversions"]].sum().reset_index()
                         .assign(controller=name, scenario=label))
    print(f"  (b) 급락 캠페인 {shocked}, {SHOCK_DAY}일차부터 전환 {SHOCK_DROP:.0%} 제거")
    show(pd.DataFrame(rows), "exp4_shock.csv")
    daily = pd.concat(daily)
    daily["cpa_x1000"] = 1000 * daily.spend / daily.conversions

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for name, col in zip(makers, SERIES):
        t = daily[(daily.controller == name) & (daily.scenario == "shock")]
        a1.plot(t.day, t.spend, "-o", color=col, lw=2, ms=4, label=name)
        a2.plot(t.day, t.cpa_x1000, "-o", color=col, lw=2, ms=4, label=name)
    for a in (a1, a2):
        a.axvline(SHOCK_DAY - 0.5, color=INK2, ls="--", lw=1)
        a.set_xticks(SIM_DAYS)
    axes(a1, "Shocked campaigns: daily spend", "day", "spend")
    axes(a2, "Shocked campaigns: daily CPA", "day", "CPA (x1000)")
    a2.legend(frameon=False, labelcolor=INK2, loc="upper left")
    save(fig, "exp4_shock.png")


# ---------------------------------------------------------------- 실험 5

FEATURE_MODELS = {
    "cost_only": "no model (score = 1 / cost)",
    "base": "base MLP",
    "+history": "+ user history",
    "+history+hour+cost": "+ history + hour + cost",
    "lasttouch": "base, last-touch label",
    "+history lasttouch": "+ history, last-touch label",
    "oracle": "oracle (true label)",
}


def exp5() -> None:
    print("\n[실험 5] 모델 개선 -> 제어 성과 (PID, 확률 크기가 달라도 스스로 맞춘다)")
    metrics = pd.read_csv(REPORTS / "exp5_model_metrics.csv")
    auc = metrics[metrics.split == "test"].set_index("variant").auc
    rows = []
    for pred in FEATURE_MODELS:
        m = market(pred, "preds_features.parquet")
        _, s = run(m, CONTROLLERS["pid"])
        rows.append({"model": pred, "auc": auc.get(pred, np.nan), **s})
    table = pd.DataFrame(rows)
    base = table.set_index("model").loc["base"]
    table["conv_vs_base"] = table.conversions / base.conversions - 1
    lo, hi = table.set_index("model").conversions[["cost_only", "oracle"]]
    table["share_of_gap"] = (table.conversions - lo) / (hi - lo)   # 바닥(0) ~ 천장(1) 중 위치
    for col in ("conv_label_sum", "conv_last_touch"):
        table[col + "_vs_base"] = table[col] / base[col] - 1
    show(table[["model", "auc", "conversions", "conv_vs_base", "share_of_gap", "impr_per_conversion",
                "conv_label_sum", "conv_label_sum_vs_base", "conv_last_touch", "conv_last_touch_vs_base",
                "cpa_x1000", "budget_used", "pacing_mae_pp", "early_depletion"]], "exp5_features.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    t = table.iloc[::-1]
    gain = 100 * t.conv_vs_base
    labels = [FEATURE_MODELS[m] + (f"  (AUC {a:.3f})" if pd.notna(a) else "") for m, a in zip(t.model, t.auc)]
    bars = ax.barh(labels, gain, color=[SERIES[0] if g >= 0 else SERIES[1] for g in gain], height=0.6)
    for b, g in zip(bars, gain):
        ax.text(g, b.get_y() + b.get_height() / 2, f" {g:+.1f}% " if g else " base ",
                va="center", ha="left" if g >= 0 else "right", color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(gain.min() - 3, gain.max() + 3)
    axes(ax, "Unique conversions vs base MLP (PID, same budget)", "conversions vs base (%)", "")
    ax.grid(axis="y", visible=False)
    save(fig, "exp5_features.png")


EXPERIMENTS = {"1": exp1, "2": exp2, "3": exp3, "4": exp4, "5": exp5}

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for key in sys.argv[1:] or EXPERIMENTS:
        EXPERIMENTS[key]()
