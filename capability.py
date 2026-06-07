import importlib.util
import numpy as np
import pandas as pd
from typing import Optional, Tuple

#02
# 로컬 statistics.py를 명시적으로 로드 (stdlib statistics 모듈과 이름 충돌 방지)
_spec = importlib.util.spec_from_file_location(
    "statistics_local",
    __file__.replace("capability.py", "statistics.py"),
)
assert _spec is not None and _spec.loader is not None, \
    "statistics.py를 찾을 수 없습니다. capability.py와 같은 폴더에 있는지 확인하세요."
_stats_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stats_mod)  # type: ignore[union-attr]
calc_unbiased_const = _stats_mod.calc_unbiased_const


# === GRADE TABLE ===
# (lower_bound, grade, label_ko, action_ko)
_GRADE_RULES = [
    (1.67, 0, "등급 0 (특급)",   "현 공정 유지 — 관리도 주기 완화 검토 가능"),
    (1.33, 1, "등급 1 (우수)",   "현 공정 유지 — 공정 변동 지속 모니터링"),
    (1.00, 2, "등급 2 (보통)",   "공정 개선 검토 — 변동 원인 분석 시작"),
    (0.67, 3, "등급 3 (불량)",   "공정 개선 필요 — 즉시 원인 조사 및 시정"),
    (0.00, 4, "등급 4 (부적합)", "즉시 생산 중단 — 공정 전면 재검토"),
]


def _grade(index_value: Optional[float]) -> Tuple[Optional[int], str, str]:
    """Return (grade_number, grade_label, action) for a capability index."""
    if index_value is None or np.isnan(index_value):
        return (None, "N/A", "단측 규격 — 해당 지수 적용 불가")
    for threshold, grade, label, action in _GRADE_RULES:
        if index_value >= threshold:
            return (grade, label, action)
    return (4, "등급 4 (부적합)", "즉시 생산 중단 — 공정 전면 재검토")


def calc_sigma_within(df: pd.DataFrame) -> dict:
    """
    Estimate within-subgroup standard deviation (σ̂_within = s_p / c4(d)).

    The pooled standard deviation combines each subgroup's sample variance
    weighted by its degrees of freedom, then corrects the bias introduced by
    the chi distribution via the c4 unbiasing constant.

    Parameters
    ----------
    df : pd.DataFrame
        Each row = one subgroup; each column = one measurement.
        Non-numeric columns and NaN cells are ignored.

    Returns
    -------
    dict
        sigma_within : float  — unbiased within-subgroup σ estimate
        s_pooled     : float  — pooled std dev (before c4 correction)
        c4           : float  — c4 constant used
        subgroup_size: int    — modal subgroup size (d)
    """
    numeric = df.select_dtypes(include="number")

    subgroup_sizes = []
    ss_sum = 0.0   # Σ (n_i - 1) * s_i²
    df_sum = 0     # Σ (n_i - 1)

    for _, row in numeric.iterrows():
        vals = row.dropna().values
        n_i = len(vals)
        if n_i < 2:
            continue
        subgroup_sizes.append(n_i)
        ss_sum += (n_i - 1) * np.var(vals, ddof=1)
        df_sum += n_i - 1

    if df_sum == 0:
        raise ValueError("No subgroup has n ≥ 2 — cannot compute σ_within.")

    s_pooled = np.sqrt(ss_sum / df_sum)

    # Use the modal subgroup size as the representative d
    counts = {}
    for s in subgroup_sizes:
        counts[s] = counts.get(s, 0) + 1
    d = max(counts, key=counts.get)

    c4_val = calc_unbiased_const("c4", d)
    sigma_within = s_pooled / c4_val

    return {
        "sigma_within":  float(sigma_within),
        "s_pooled":      float(s_pooled),
        "c4":            float(c4_val),
        "subgroup_size": int(d),
    }


def calc_sigma_overall(df: pd.DataFrame) -> dict:
    """
    Estimate overall (long-term) standard deviation (σ̂_overall = s / c4(n)).

    Flattens all measurements into a single sample, computes the ordinary
    sample standard deviation, then removes the chi-distribution bias via
    c4(n_total).

    Parameters
    ----------
    df : pd.DataFrame
        Each row = one subgroup; each column = one measurement.
        Non-numeric columns and NaN cells are ignored.

    Returns
    -------
    dict
        sigma_overall : float — unbiased overall σ estimate
        s_overall     : float — raw sample std dev (before c4 correction)
        c4            : float — c4 constant used
        n_total       : int   — total number of measurements
    """
    numeric = df.select_dtypes(include="number")
    all_vals = numeric.values.flatten()
    all_vals = all_vals[~pd.isna(all_vals)].astype(float)

    n_total = len(all_vals)
    if n_total < 2:
        raise ValueError("Need at least 2 measurements to compute σ_overall.")

    s_overall = float(np.std(all_vals, ddof=1))
    c4_val = calc_unbiased_const("c4", n_total) if n_total <= 25 else \
             _c4_approx(n_total)
    sigma_overall = s_overall / c4_val

    return {
        "sigma_overall": float(sigma_overall),
        "s_overall":     float(s_overall),
        "c4":            float(c4_val),
        "n_total":       int(n_total),
    }


def _c4_approx(n: int) -> float:
    """c4 approximation for n > 25 via Lanczos gamma."""
    def _lgamma(x):
        # Stirling's series — accurate enough for large x
        return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) + \
               1 / (12 * x) - 1 / (360 * x ** 3)

    log_c4 = 0.5 * np.log(2 / (n - 1)) + _lgamma(n / 2) - _lgamma((n - 1) / 2)
    return float(np.exp(log_c4))


def calc_capability_indices(
    x_bar: float,
    sigma_within: float,
    sigma_overall: float,
    USL: Optional[float] = None,
    LSL: Optional[float] = None,
) -> dict:
    """
    Compute process capability (Cp / Cpk) and performance (Pp / Ppk) indices.

    One-sided handling
    ------------------
    * Both USL and LSL present → bilateral indices.
    * Only USL present         → Cp/Pp = None; Cpk = CPU, Ppk = PPU.
    * Only LSL present         → Cp/Pp = None; Cpk = CPL, Ppk = PPL.

    Grade scheme (based on the primary index: min(Cpk, Ppk) when both exist)
    --------------------------------------------------------------------------
    ≥ 1.67 → Grade 0  |  1.33–1.67 → Grade 1  |  1.00–1.33 → Grade 2
    0.67–1.00 → Grade 3  |  < 0.67 → Grade 4

    Parameters
    ----------
    x_bar         : float       — overall process mean (x̄̄)
    sigma_within  : float       — unbiased within-subgroup σ (from calc_sigma_within)
    sigma_overall : float       — unbiased overall σ (from calc_sigma_overall)
    USL           : float|None  — upper specification limit
    LSL           : float|None  — lower specification limit

    Returns
    -------
    dict with keys:
        Cp, Cpk, Pp, Ppk        : float|None
        CPU, CPL, PPU, PPL      : float|None  (one-sided components)
        grade_Cpk               : int|None
        grade_Ppk               : int|None
        grade_label             : str
        action                  : str
        x_bar, sigma_within, sigma_overall : echoed back for convenience
    """
    if USL is None and LSL is None:
        raise ValueError("At least one of USL or LSL must be provided.")

    # --- Short-hand helpers -------------------------------------------------
    def _ratio(span, sigma):
        return float(span / (6 * sigma)) if span is not None else None

    def _one_sided(limit, sigma, upper=True):
        if limit is None:
            return None
        dist = (limit - x_bar) if upper else (x_bar - limit)
        return float(dist / (3 * sigma))

    # --- Bilateral Cp / Pp --------------------------------------------------
    span = (USL - LSL) if (USL is not None and LSL is not None) else None
    Cp  = _ratio(span, sigma_within)
    Pp  = _ratio(span, sigma_overall)

    # --- One-sided components -----------------------------------------------
    CPU = _one_sided(USL, sigma_within, upper=True)
    CPL = _one_sided(LSL, sigma_within, upper=False)
    PPU = _one_sided(USL, sigma_overall, upper=True)
    PPL = _one_sided(LSL, sigma_overall, upper=False)

    # --- Cpk / Ppk ----------------------------------------------------------
    if USL is not None and LSL is not None:
        Cpk = float(min(CPU, CPL))
        Ppk = float(min(PPU, PPL))
    elif USL is not None:
        Cpk = CPU
        Ppk = PPU
    else:
        Cpk = CPL
        Ppk = PPL

    # --- Grade (use the more conservative: Cpk) -----------------------------
    grade_num_cpk, grade_label, action = _grade(Cpk)
    grade_num_ppk, _, _               = _grade(Ppk)

    return {
        # Primary indices
        "Cp":              Cp,
        "Cpk":             Cpk,
        "Pp":              Pp,
        "Ppk":             Ppk,
        # One-sided components
        "CPU":             CPU,
        "CPL":             CPL,
        "PPU":             PPU,
        "PPL":             PPL,
        # Grade info
        "grade_Cpk":       grade_num_cpk,
        "grade_Ppk":       grade_num_ppk,
        "grade_label":     grade_label,
        "action":          action,
        # Echoed inputs
        "x_bar":           float(x_bar),
        "sigma_within":    float(sigma_within),
        "sigma_overall":   float(sigma_overall),
        "USL":             USL,
        "LSL":             LSL,
    }
