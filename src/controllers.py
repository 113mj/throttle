"""컨트롤러: 15분 구간마다 캠페인별 노출 기준을 정한다.

노출 점수(score)는 시뮬레이터가 정한다 (기본: pCVR / 비용 = 비용 1당 기대 전환).
컨트롤러는 "점수 >= 임계값" 의 임계값만 조정한다.

공통 인터페이스
  start_day()            하루 시작
  decide(score) -> mask  이번 구간 노출들을 보여줄지
  update(step)           구간이 끝난 뒤 소진 상황을 보고 다음 구간 기준을 조정
  observe(shown)         이번 구간에 보여준 노출들 (성과 피드백을 쓰는 컨트롤러만 사용)
  end_day(now)           하루 끝 (now = 그 시점의 timestamp)
  threshold              현재 임계값 (기록용)
"""
from dataclasses import dataclass

import numpy as np

TAU_RANGE = 1e3   # PID 임계값은 초기값의 1/1000 ~ 1000배 안에서 움직인다


@dataclass
class Step:
    k: int                # 구간 번호 0..95
    spent_frac: float     # 오늘 누적 소진 / 예산
    target_frac: float    # 구간 k 끝 시점 목표 누적 소진 비율
    target_next: float    # 다음 구간 목표 소진 증가분 (예산 대비 비율)
    remaining: float      # 남은 예산
    score: np.ndarray     # 이번 구간 노출들의 점수
    cost: np.ndarray      # 이번 구간 노출들의 비용


class Controller:
    name = "base"
    threshold = 0.0

    def start_day(self) -> None:
        pass

    def decide(self, score: np.ndarray) -> np.ndarray:
        return score >= self.threshold

    def update(self, step: Step) -> None:
        pass

    def observe(self, shown: dict) -> None:
        pass

    def end_day(self, now: float | None = None) -> None:
        pass


class Greedy(Controller):
    """기준 없이 전부 보여준다 (예산이 바닥나면 중단). 참고용."""
    name = "greedy"


class FixedThreshold(Controller):
    """하루 종일 같은 임계값."""
    name = "fixed"

    def __init__(self, tau: float):
        self.threshold = tau


class Pacing(FixedThreshold):
    """고정 임계값을 넘는 노출 중 일부만 확률적으로 보여준다 (probabilistic throttling).

    통과 비율 r = 다음 구간 계획 소진 / 이번 구간 후보 비용
    계획 소진 = 남은 예산 x (다음 구간 목표 증가분 / 남은 목표 비율)
    """
    name = "pacing"

    def __init__(self, tau: float, seed: int = 0):
        super().__init__(tau)
        self.rng = np.random.default_rng(seed)
        self.rate = 1.0

    def start_day(self) -> None:
        self.rate = 1.0

    def decide(self, score: np.ndarray) -> np.ndarray:
        return (score >= self.threshold) & (self.rng.random(len(score)) < self.rate)

    def update(self, step: Step) -> None:
        eligible = step.cost[step.score >= self.threshold].sum()
        left = 1.0 - step.target_frac
        if left <= 0 or eligible <= 0:
            self.rate = 1.0
            return
        planned = step.remaining * step.target_next / left
        self.rate = float(np.clip(planned / eligible, 0.0, 1.0))


class PID(Controller):
    """목표 누적 소진 대비 실제 누적 소진의 오차로 임계값을 조정한다.

    e = 목표 - 실제 (양수 = 덜 쓰고 있음 -> 임계값을 낮춘다)
    log(tau) = base - (Kp*e + Ki*sum(e) + Kd*(e - e_prev))
    배율로 움직이도록 로그 스케일에서 조정. 하루 평균 log(tau) 를 다음 날 base 로 넘긴다.
    """
    name = "pid"

    def __init__(self, tau0: float, kp: float = 4.0, ki: float = 0.2, kd: float = 0.0,
                 i_max: float = 5.0):
        self.base = np.log(tau0)
        self.lo, self.hi = np.log(tau0 / TAU_RANGE), np.log(tau0 * TAU_RANGE)
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self.log_tau = self.base

    @property
    def threshold(self) -> float:
        return float(np.exp(self.log_tau))

    def start_day(self) -> None:
        self.integral, self.prev_e = 0.0, 0.0
        self.log_tau = self.base
        self.day_log_taus = []

    def update(self, step: Step) -> None:
        e = step.target_frac - step.spent_frac
        self.integral = float(np.clip(self.integral + e, -self.i_max, self.i_max))
        u = self.kp * e + self.ki * self.integral + self.kd * (e - self.prev_e)
        self.prev_e = e
        self.log_tau = float(np.clip(self.base - u, self.lo, self.hi))
        self.day_log_taus.append(self.log_tau)

    def end_day(self, now: float | None = None) -> None:
        if self.day_log_taus:
            self.base = float(np.mean(self.day_log_taus))


class FeedbackPID(PID):
    """PID (안쪽 루프, 15분) + 효율 하한선 + 성과 피드백 (바깥쪽 루프, 하루 1번).

    - 하한선: 점수가 floor 미만인 노출은 PID 임계값과 관계없이 사지 않는다 (목표 CPA).
      하한선이 없으면 점수를 일괄로 깎아도 PID 가 임계값을 같이 낮춰 지출이 그대로다.
    - 피드백: 매일 끝에 최근 window 일 동안 보여준 노출로
        비율 = 관측된 전환 / 기대 관측 전환
      을 계산해 다음 날 점수에 곱한다.
        mode="delay": 기대 = Σ pCVR x F(지금 - 노출 시각)  (F = 노출 후 전환 지연의 누적분포)
        mode="naive": 기대 = Σ pCVR                        (지연 무시)
        mode="none" : 보정 안 함
      비율은 의사 관측치 prior 로 1 쪽으로 당겨 잡음을 줄인다.
    - baseline_days: 주면 처음 그 일수 동안은 보정하지 않고, 그때의 비율을 기준값으로 삼아
      이후에는 (오늘 비율 / 기준값) 을 쓴다. 고른 노출만 보면 생기는 선택 편향(승자의 저주)처럼
      평소에도 일정하게 어긋나는 오차를 지우고 "평소 대비 변화"만 본다.
    """
    name = "feedback_pid"

    def __init__(self, tau0: float, floor: float, mode: str = "delay", delay_cdf=None,
                 window_days: int = 4, prior: float = 20.0, r_clip=(0.2, 2.0),
                 baseline_days: int | None = None, **pid_kw):
        super().__init__(tau0, **pid_kw)
        self.floor, self.mode, self.cdf = floor, mode, delay_cdf
        self.baseline_days, self.r_ref, self.days_seen = baseline_days, None, 0
        self.window, self.prior, self.r_clip = window_days * 86400, prior, r_clip
        self.ratio = 1.0
        self.ratios = []          # 하루 끝마다 계산한 비율 (기록용)
        self.hist = {k: [] for k in ("ts", "p", "label", "conv_ts")}

    def decide(self, score: np.ndarray) -> np.ndarray:
        s = score * self.ratio
        return (s >= self.threshold) & (s >= self.floor)

    def observe(self, shown: dict) -> None:
        for k in self.hist:
            self.hist[k].append(shown[k])

    def end_day(self, now: float | None = None) -> None:
        super().end_day(now)
        if self.mode == "none" or now is None:
            return
        h = {k: np.concatenate(v) for k, v in self.hist.items()}
        recent = h["ts"] >= now - self.window
        observed = ((h["label"] == 1) & (h["conv_ts"] <= now))[recent].sum()
        p = h["p"][recent]
        if self.mode == "delay":
            expected = (p * self.cdf(now - h["ts"][recent])).sum()
        else:
            expected = p.sum()
        r = (observed + self.prior) / (expected + self.prior)
        self.days_seen += 1
        if self.baseline_days is not None:
            if self.days_seen <= self.baseline_days:
                self.r_ref = r          # 기준 기간의 마지막 날 비율 (그 전 며칠의 노출을 다 포함)
                self.ratios.append(1.0)
                return
            r = r / self.r_ref
        self.ratio = float(np.clip(r, *self.r_clip))
        self.ratios.append(self.ratio)
