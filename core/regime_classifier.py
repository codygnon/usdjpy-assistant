"""
Score-based market regime classifier for the eligibility-level router.

Five regimes:
    momentum           – V44-favorable: strong directional move underway
    mean_reversion     – V14-favorable: compressed, non-aligned, low ADX
    breakout           – London V2-favorable: fresh expansion from compression
    post_breakout_trend – continuation after breakout has already occurred
    ambiguous          – no regime scores clearly above the rest

Stability layers (applied in order):
    Layer 1 – Soft scoring: smooth ramps replace binary thresholds.
              Each component contributes 0.0–1.0 continuously.
              Toggle: soft_scoring_enabled (default False).

    Layer 2 – Score EMA smoothing: exponential moving average of per-regime
              scores across bars.  Prevents single-bar noise from flipping
              the winner.
              Toggle: score_ema_enabled (default False).
              Param:  score_ema_alpha (default 0.3, ~3-bar half-life on M5).

    Layer 3 – Schmitt-trigger hysteresis: separate enter/exit thresholds.
              A regime activates when smoothed score > enter_threshold and
              deactivates when it drops below exit_threshold.
              Toggle: schmitt_enabled (default False).
              Params: schmitt_enter_threshold, schmitt_exit_threshold.

    Safety net – Dwell-time: even after all layers, a regime must hold for
              at least dwell_bars before transitioning.  Acts as a final
              backstop, not the primary stability mechanism.

Advanced feature experiment:
    A narrow V44 exhaustion penalty can optionally attenuate the momentum
    score when BOTH efficiency is poor and trend decay is high. This is a
    targeted, multiplicative downgrade only; it does not rewrite other
    regime scores.

h1_aligned contract:
    Callers must provide a SIDE-AWARE boolean: True when the H1 EMA
    structure (fast vs slow) agrees with the M5 slope direction.
    e.g. h1_fast > h1_slow AND m5_slope > 0, OR h1_fast < h1_slow AND
    m5_slope < 0.  This is NOT simply "bullish" — a short-side momentum
    bar with h1_fast < h1_slow is also "aligned."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.regime_features import RegimeFeatures


# ── Soft ramp helper ────────────────────────────────────────────────

def _ramp(value: float, low: float, high: float) -> float:
    """Smooth linear ramp: 0.0 at low, 1.0 at high, clamped."""
    if high <= low:
        return 1.0 if value >= low else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _inv_ramp(value: float, low: float, high: float) -> float:
    """Inverse ramp: 1.0 at low, 0.0 at high, clamped."""
    return 1.0 - _ramp(value, low, high)


def _trapezoid(value: float, lo_edge: float, lo_flat: float,
               hi_flat: float, hi_edge: float) -> float:
    """Trapezoid: ramps up from lo_edge→lo_flat, flat 1.0, ramps down hi_flat→hi_edge."""
    if value <= lo_edge or value >= hi_edge:
        return 0.0
    if value < lo_flat:
        return _ramp(value, lo_edge, lo_flat)
    if value > hi_flat:
        return _inv_ramp(value, hi_flat, hi_edge)
    return 1.0


# ── Threshold defaults (data-driven from regime_threshold_analysis) ──

@dataclass
class RegimeThresholds:
    # Momentum (V44)
    momentum_adx_min: float = 25.0
    momentum_slope_min: float = 0.9
    momentum_bb_pctile_min: float = 0.30
    momentum_bb_exp_rate_max: Optional[float] = None  # v6 experiment: upper bound (replaces bb_pctile)
    momentum_bb_exp_rate_min: Optional[float] = None  # v6 experiment: lower bound (band mode)

    # Mean reversion (V14)
    mr_adx_max: float = 29.0
    mr_bb_pctile_max: float = 0.40

    # Breakout (London V2)
    bo_bb_pctile_max: float = 0.75     # cap over-extension
    bo_adx_min: float = 12.0           # not dead flat
    bo_adx_max: float = 45.0           # not already trending hard
    bo_slope_min: float = 0.3          # moderate movement, live breakout

    # Post-breakout trend
    pbt_bb_pctile_min: float = 0.60    # expansion already underway
    pbt_adx_min: float = 20.0          # trend confirmed

    # Winner selection
    min_margin: float = 0.5            # score margin to beat runner-up
    dwell_bars: int = 5                # hold regime for at least N bars

    # ── Layer 1: Soft scoring ramp widths ──
    # Each ramp extends ± half-width around the threshold center.
    # Set soft_scoring_enabled=False to revert to binary 0/1 scoring.
    soft_scoring_enabled: bool = False
    soft_ramp_half_width_adx: float = 5.0       # ADX ramps over ±5 around center
    soft_ramp_half_width_slope: float = 0.4      # slope ramps over ±0.4
    soft_ramp_half_width_bb_pctile: float = 0.15 # bb_pctile ramps over ±0.15

    # ── Layer 2: Score EMA smoothing ──
    score_ema_enabled: bool = False
    score_ema_alpha: float = 0.3       # ~3-bar half-life on M5

    # ── Layer 3: Schmitt-trigger hysteresis ──
    schmitt_enabled: bool = False
    schmitt_enter_threshold: float = 2.8   # smoothed score must exceed to activate
    schmitt_exit_threshold: float = 2.0    # smoothed score must drop below to deactivate

    # ── Advanced features: narrow V44 exhaustion penalty ──
    # Safe baseline: disabled unless explicitly enabled by experiment code.
    # Only efficiency_ratio and trend_decay_rate are used in the decision path.
    features_enabled: bool = False
    feat_v44_exhaustion_enabled: bool = False
    feat_er_threshold: float = 0.35           # ER must be BELOW this to trigger
    feat_decay_threshold: float = 0.40        # decay must be ABOVE this to trigger
    feat_momentum_attenuator: float = 0.75    # multiply momentum score by this when triggered


# ── Score result ─────────────────────────────────────────────────────

@dataclass
class RegimeScores:
    momentum: float = 0.0
    mean_reversion: float = 0.0
    breakout: float = 0.0
    post_breakout_trend: float = 0.0

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "momentum": self.momentum,
            "mean_reversion": self.mean_reversion,
            "breakout": self.breakout,
            "post_breakout_trend": self.post_breakout_trend,
        }


@dataclass
class RegimeResult:
    label: str
    scores: RegimeScores
    margin: float              # gap between winner and runner-up
    held_by_dwell: bool        # True if label was kept due to hysteresis
    raw_scores: Optional[RegimeScores] = None    # pre-EMA scores (for diagnostics)
    smoothed_scores: Optional[RegimeScores] = None  # post-EMA scores (for diagnostics)


# ── Layer 1: Soft scoring functions ─────────────────────────────────
# Each component returns 0.0–1.0 continuously (soft) or 0/1 (hard).
# Max score = 4 for all regimes.


def _momentum_score(
    adx: float,
    m5_slope_abs: float,
    h1_aligned: bool,
    bb_width_pctile: float,
    th: RegimeThresholds,
    *,
    bb_width_exp_rate: Optional[float] = None,
) -> float:
    """4 components: strong ADX, high slope, H1 aligned, volatility component."""
    s = 0.0
    if th.soft_scoring_enabled:
        hw_adx = th.soft_ramp_half_width_adx
        hw_slope = th.soft_ramp_half_width_slope
        hw_bb = th.soft_ramp_half_width_bb_pctile

        # ADX: higher is more momentum-like
        s += _ramp(adx, th.momentum_adx_min - hw_adx, th.momentum_adx_min + hw_adx)
        # Slope: higher is more momentum-like
        s += _ramp(m5_slope_abs, th.momentum_slope_min - hw_slope, th.momentum_slope_min + hw_slope)
    else:
        if adx >= th.momentum_adx_min:
            s += 1.0
        if m5_slope_abs >= th.momentum_slope_min:
            s += 1.0

    # H1 alignment: inherently binary
    if h1_aligned:
        s += 1.0

    # Volatility component
    if th.momentum_bb_exp_rate_max is not None:
        if bb_width_exp_rate is not None:
            lo = th.momentum_bb_exp_rate_min if th.momentum_bb_exp_rate_min is not None else -float("inf")
            if lo <= bb_width_exp_rate <= th.momentum_bb_exp_rate_max:
                s += 1.0
    else:
        if th.soft_scoring_enabled:
            hw_bb = th.soft_ramp_half_width_bb_pctile
            s += _ramp(bb_width_pctile, th.momentum_bb_pctile_min - hw_bb, th.momentum_bb_pctile_min + hw_bb)
        else:
            if bb_width_pctile >= th.momentum_bb_pctile_min:
                s += 1.0
    return s


def _mean_reversion_score(
    adx: float,
    h1_aligned: bool,
    bb_width_pctile: float,
    bb_regime: str,
    th: RegimeThresholds,
) -> float:
    """4 components: ranging BB, low ADX, compressed width, H1 disagrees with M5."""
    s = 0.0

    # BB regime: inherently categorical/binary
    if bb_regime == "ranging":
        s += 1.0

    if th.soft_scoring_enabled:
        hw_adx = th.soft_ramp_half_width_adx
        hw_bb = th.soft_ramp_half_width_bb_pctile
        # ADX: lower is more mean-reversion-like (inverse ramp)
        s += _inv_ramp(adx, th.mr_adx_max - hw_adx, th.mr_adx_max + hw_adx)
        # BB width: lower is more compressed (inverse ramp)
        s += _inv_ramp(bb_width_pctile, th.mr_bb_pctile_max - hw_bb, th.mr_bb_pctile_max + hw_bb)
    else:
        if adx < th.mr_adx_max:
            s += 1.0
        if bb_width_pctile < th.mr_bb_pctile_max:
            s += 1.0

    # H1 disagrees: inherently binary
    if not h1_aligned:
        s += 1.0
    return s


def _breakout_score(
    adx: float,
    m5_slope_abs: float,
    bb_width_pctile: float,
    bb_regime: str,
    th: RegimeThresholds,
) -> float:
    """4 components: ranging BB, compressed width, moderate ADX, moderate slope."""
    s = 0.0

    # BB regime: inherently categorical/binary
    if bb_regime == "ranging":
        s += 1.0

    if th.soft_scoring_enabled:
        hw_bb = th.soft_ramp_half_width_bb_pctile
        hw_adx = th.soft_ramp_half_width_adx
        hw_slope = th.soft_ramp_half_width_slope

        # BB width: compressed, not over-extended (inverse ramp)
        s += _inv_ramp(bb_width_pctile, th.bo_bb_pctile_max - hw_bb, th.bo_bb_pctile_max + hw_bb)
        # ADX: moderate range (trapezoid)
        s += _trapezoid(adx,
                        th.bo_adx_min - hw_adx, th.bo_adx_min + hw_adx,
                        th.bo_adx_max - hw_adx, th.bo_adx_max + hw_adx)
        # Slope: moderate range (trapezoid)
        s += _trapezoid(m5_slope_abs,
                        th.bo_slope_min - hw_slope, th.bo_slope_min + hw_slope,
                        th.momentum_slope_min - hw_slope, th.momentum_slope_min + hw_slope)
    else:
        if bb_width_pctile < th.bo_bb_pctile_max:
            s += 1.0
        if th.bo_adx_min <= adx <= th.bo_adx_max:
            s += 1.0
        if th.bo_slope_min <= m5_slope_abs < th.momentum_slope_min:
            s += 1.0
    return s


def _post_breakout_trend_score(
    adx: float,
    h1_aligned: bool,
    bb_width_pctile: float,
    bb_regime: str,
    th: RegimeThresholds,
) -> float:
    """4 components: expanded BB width, trending regime, strong ADX, H1 aligned."""
    s = 0.0

    if th.soft_scoring_enabled:
        hw_bb = th.soft_ramp_half_width_bb_pctile
        hw_adx = th.soft_ramp_half_width_adx

        # BB width: expanded (ramp up)
        s += _ramp(bb_width_pctile, th.pbt_bb_pctile_min - hw_bb, th.pbt_bb_pctile_min + hw_bb)
        # ADX: strong trend confirmed (ramp up)
        s += _ramp(adx, th.pbt_adx_min - hw_adx, th.pbt_adx_min + hw_adx)
    else:
        if bb_width_pctile >= th.pbt_bb_pctile_min:
            s += 1.0
        if adx >= th.pbt_adx_min:
            s += 1.0

    # BB regime: inherently categorical/binary
    if bb_regime == "trending":
        s += 1.0
    # H1 alignment: inherently binary
    if h1_aligned:
        s += 1.0
    return s


# ── Advanced feature adjustments ────────────────────────────────────

def _apply_feature_adjustments(
    scores: RegimeScores,
    features: "RegimeFeatures",
    th: RegimeThresholds,
) -> RegimeScores:
    """Targeted multiplicative attenuation of the momentum score.

    Only fires when BOTH conditions are met simultaneously:
      - efficiency_ratio < feat_er_threshold  (price is churning)
      - trend_decay_rate > feat_decay_threshold  (slope is collapsing)

    When triggered, multiplies only momentum by the configured attenuator.
    This preserves the base score shape — a 3.2 momentum score becomes 2.4,
    not 0.0.

    All other regimes are left untouched.
    When the gate is not triggered, scores pass through unchanged.
    """
    if not th.feat_v44_exhaustion_enabled:
        return scores

    er = features.efficiency_ratio
    decay = features.trend_decay_rate

    # Gate: both must be bad together
    exhaustion_triggered = (er < th.feat_er_threshold and decay > th.feat_decay_threshold)

    if not exhaustion_triggered:
        return scores

    d = scores.as_dict
    d["momentum"] *= th.feat_momentum_attenuator

    return RegimeScores(**d)


# ── Stateless classification (single bar) ───────────────────────────

def classify_bar(
    adx: float,
    m5_slope_abs: float,
    h1_aligned: bool,
    bb_width_pctile: float,
    bb_regime: str,
    th: Optional[RegimeThresholds] = None,
    *,
    bb_width_exp_rate: Optional[float] = None,
    features: Optional["RegimeFeatures"] = None,
) -> tuple[str, RegimeScores, float]:
    """Classify one bar without hysteresis.

    Returns (label, scores, margin).
    Uses soft scoring if th.soft_scoring_enabled (Layer 1).
    Applies feature adjustments if features is provided and th.features_enabled.
    """
    if th is None:
        th = RegimeThresholds()

    scores = RegimeScores(
        momentum=_momentum_score(
            adx, m5_slope_abs, h1_aligned, bb_width_pctile, th,
            bb_width_exp_rate=bb_width_exp_rate,
        ),
        mean_reversion=_mean_reversion_score(adx, h1_aligned, bb_width_pctile, bb_regime, th),
        breakout=_breakout_score(adx, m5_slope_abs, bb_width_pctile, bb_regime, th),
        post_breakout_trend=_post_breakout_trend_score(adx, h1_aligned, bb_width_pctile, bb_regime, th),
    )

    # Apply advanced feature adjustments
    if features is not None and th.features_enabled:
        scores = _apply_feature_adjustments(scores, features, th)

    ranked = sorted(scores.as_dict.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    runner_up_score = ranked[1][1]
    margin = best_score - runner_up_score

    if margin >= th.min_margin and best_score >= 2.0:
        label = best_label
    else:
        label = "ambiguous"

    return label, scores, margin


# ── Stateful classifier with all stability layers ───────────────────

_REGIME_KEYS = ("momentum", "mean_reversion", "breakout", "post_breakout_trend")


class RegimeClassifier:
    """Bar-by-bar classifier with three stability layers + dwell-time safety net.

    Feed bars sequentially via ``update()``.

    Layer 1 (soft scoring): controlled by th.soft_scoring_enabled.
        Applied inside the scoring functions — no state needed.

    Layer 2 (score EMA): controlled by th.score_ema_enabled / score_ema_alpha.
        Maintains per-regime EMA of raw scores.

    Layer 3 (Schmitt trigger): controlled by th.schmitt_enabled.
        Uses separate enter/exit thresholds on smoothed scores.

    Safety net (dwell-time): th.dwell_bars.
        After all layers, a regime still holds for at least N bars.
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.th = thresholds or RegimeThresholds()
        self._current_label: str = "ambiguous"
        self._bars_since_change: int = 0
        # Layer 2: EMA state
        self._ema_scores: dict[str, float] = {k: 0.0 for k in _REGIME_KEYS}
        self._ema_initialized: bool = False

    @property
    def current_label(self) -> str:
        return self._current_label

    def reset(self) -> None:
        self._current_label = "ambiguous"
        self._bars_since_change = 0
        self._ema_scores = {k: 0.0 for k in _REGIME_KEYS}
        self._ema_initialized = False

    def update(
        self,
        adx: float,
        m5_slope_abs: float,
        h1_aligned: bool,
        bb_width_pctile: float,
        bb_regime: str,
        *,
        bb_width_exp_rate: Optional[float] = None,
        features: Optional["RegimeFeatures"] = None,
    ) -> RegimeResult:
        # ── Layer 1: Soft scoring + feature adjustments (inside classify_bar) ──
        raw_label, raw_scores, raw_margin = classify_bar(
            adx, m5_slope_abs, h1_aligned, bb_width_pctile, bb_regime, self.th,
            bb_width_exp_rate=bb_width_exp_rate,
            features=features,
        )

        # ── Layer 2: EMA smoothing ──
        raw_dict = raw_scores.as_dict
        if self.th.score_ema_enabled:
            alpha = self.th.score_ema_alpha
            if not self._ema_initialized:
                # Seed EMA with first bar's raw scores
                self._ema_scores = dict(raw_dict)
                self._ema_initialized = True
            else:
                for k in _REGIME_KEYS:
                    self._ema_scores[k] = alpha * raw_dict[k] + (1.0 - alpha) * self._ema_scores[k]
            smoothed = dict(self._ema_scores)
        else:
            smoothed = dict(raw_dict)

        smoothed_scores = RegimeScores(**smoothed)

        # ── Layer 3: Schmitt-trigger activation ──
        if self.th.schmitt_enabled:
            candidate_label = self._schmitt_select(smoothed)
        else:
            # Fall back to margin-based selection on smoothed scores
            ranked = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
            best_label, best_score = ranked[0]
            runner_up_score = ranked[1][1]
            margin = best_score - runner_up_score
            if margin >= self.th.min_margin and best_score >= 2.0:
                candidate_label = best_label
            else:
                candidate_label = "ambiguous"

        # Compute margin for diagnostics (on smoothed scores)
        ranked = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1]

        # ── Safety net: dwell-time ──
        self._bars_since_change += 1
        held = False

        if candidate_label != self._current_label:
            if self._bars_since_change >= self.th.dwell_bars:
                self._current_label = candidate_label
                self._bars_since_change = 0
            else:
                held = True

        return RegimeResult(
            label=self._current_label,
            scores=smoothed_scores,
            margin=margin,
            held_by_dwell=held,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
        )

    def _schmitt_select(self, smoothed: dict[str, float]) -> str:
        """Schmitt-trigger regime selection.

        If current regime's smoothed score is still above exit_threshold,
        keep it (sticky).  Otherwise, switch to any regime whose smoothed
        score exceeds enter_threshold (pick highest).  If none qualifies,
        go ambiguous.
        """
        enter = self.th.schmitt_enter_threshold
        exit_ = self.th.schmitt_exit_threshold

        # Current regime still above exit threshold? Keep it.
        if (self._current_label != "ambiguous"
                and self._current_label in smoothed
                and smoothed[self._current_label] >= exit_):
            return self._current_label

        # Find best candidate that exceeds enter threshold
        candidates = [(k, v) for k, v in smoothed.items() if v >= enter]
        if candidates:
            best = max(candidates, key=lambda kv: kv[1])
            return best[0]

        return "ambiguous"
