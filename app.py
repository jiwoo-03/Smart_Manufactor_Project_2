import numpy as np
import pandas as pd
from scipy import stats

#01.
# === LOOKUP TABLE: n=2~25 ===
_CONST_TABLE = {
    #  n:  (c4,      d2,     d3,     d4,     A2,     A3,     B3,     B4,     D3,     D4)
    2:  (0.7979, 1.1284, 0.8525, 0.9534, 1.8800, 2.6590, 0.0000, 3.2670, 0.0000, 3.2670),
    3:  (0.8862, 1.6926, 0.8884, 0.9594, 1.0230, 1.9540, 0.0000, 2.5680, 0.0000, 2.5740),
    4:  (0.9213, 2.0588, 0.8798, 0.9550, 0.7290, 1.6280, 0.0000, 2.2660, 0.0000, 2.2820),
    5:  (0.9400, 2.3259, 0.8641, 0.9515, 0.5770, 1.4270, 0.0000, 2.0890, 0.0000, 2.1140),
    6:  (0.9515, 2.5344, 0.8480, 0.9490, 0.4830, 1.2870, 0.0300, 1.9700, 0.0000, 2.0040),
    7:  (0.9594, 2.7044, 0.8332, 0.9466, 0.4190, 1.1820, 0.1180, 1.8820, 0.0760, 1.9240),
    8:  (0.9650, 2.8472, 0.8198, 0.9444, 0.3730, 1.0990, 0.1850, 1.8150, 0.1360, 1.8640),
    9:  (0.9693, 2.9700, 0.8078, 0.9423, 0.3370, 1.0320, 0.2390, 1.7610, 0.1840, 1.8160),
    10: (0.9727, 3.0777, 0.7971, 0.9403, 0.3080, 0.9750, 0.2840, 1.7160, 0.2230, 1.7770),
    11: (0.9754, 3.1729, 0.7873, 0.9385, 0.2850, 0.9270, 0.3210, 1.6790, 0.2560, 1.7440),
    12: (0.9776, 3.2585, 0.7785, 0.9367, 0.2660, 0.8860, 0.3540, 1.6460, 0.2830, 1.7170),
    13: (0.9794, 3.3367, 0.7704, 0.9351, 0.2490, 0.8500, 0.3820, 1.6180, 0.3070, 1.6930),
    14: (0.9810, 3.4068, 0.7630, 0.9335, 0.2350, 0.8170, 0.4060, 1.5940, 0.3280, 1.6720),
    15: (0.9823, 3.4718, 0.7562, 0.9320, 0.2230, 0.7890, 0.4280, 1.5720, 0.3470, 1.6530),
    16: (0.9835, 3.5318, 0.7499, 0.9306, 0.2120, 0.7630, 0.4480, 1.5520, 0.3630, 1.6370),
    17: (0.9845, 3.5879, 0.7441, 0.9293, 0.2030, 0.7390, 0.4660, 1.5340, 0.3780, 1.6220),
    18: (0.9854, 3.6397, 0.7386, 0.9281, 0.1940, 0.7180, 0.4820, 1.5180, 0.3910, 1.6080),
    19: (0.9862, 3.6890, 0.7335, 0.9269, 0.1870, 0.6980, 0.4970, 1.5030, 0.4030, 1.5970),
    20: (0.9869, 3.7354, 0.7287, 0.9257, 0.1800, 0.6800, 0.5100, 1.4900, 0.4150, 1.5850),
    21: (0.9876, 3.7798, 0.7242, 0.9246, 0.1730, 0.6630, 0.5230, 1.4770, 0.4250, 1.5750),
    22: (0.9882, 3.8220, 0.7199, 0.9235, 0.1670, 0.6470, 0.5340, 1.4660, 0.4340, 1.5660),
    23: (0.9887, 3.8625, 0.7159, 0.9225, 0.1620, 0.6330, 0.5450, 1.4550, 0.4430, 1.5570),
    24: (0.9892, 3.9004, 0.7121, 0.9215, 0.1570, 0.6190, 0.5550, 1.4450, 0.4510, 1.5490),
    25: (0.9896, 3.9367, 0.7084, 0.9206, 0.1530, 0.6060, 0.5650, 1.4350, 0.4590, 1.5410),
}
_COL_IDX = {"c4": 0, "d2": 1, "d3": 2, "d4": 3,
            "A2": 4, "A3": 5, "B3": 6, "B4": 7, "D3": 8, "D4": 9}


def calc_unbiased_const(const_name: str, n: int) -> float:
    """
    Return the SPC unbiased constant for subgroup size n.

    Parameters
    ----------
    const_name : str
        One of 'c4', 'd2', 'd3', 'd4', 'A2', 'A3', 'B3', 'B4', 'D3', 'D4'.
    n : int
        Subgroup size (≥ 2).

    Returns
    -------
    float
        Constant value.
    """
    if const_name not in _COL_IDX:
        raise ValueError(f"Unknown constant '{const_name}'. "
                         f"Choose from {list(_COL_IDX.keys())}.")
    if n < 2:
        raise ValueError("Subgroup size n must be ≥ 2.")

    if n <= 25:
        return _CONST_TABLE[n][_COL_IDX[const_name]]

    # Approximations for n > 25
    if const_name == "c4":
        return np.sqrt(2 / (n - 1)) * (np.math.factorial(n // 2 - 1) /
               np.math.factorial((n - 3) // 2)) / np.sqrt(np.pi) \
               if n % 2 == 1 else \
               np.sqrt(2 / (n - 1)) * (np.math.factorial(n // 2 - 1) /
               np.math.factorial((n - 2) // 2)) * np.sqrt(np.pi / 2)

    # Numerical c4 via Lanczos gamma
    def _gamma(x):
        g = 7
        p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1 - x))
        x -= 1
        a = p[0]
        t = x + g + 0.5
        for i in range(1, g + 2):
            a += p[i] / (x + i)
        return np.sqrt(2 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a

    def _c4(n_):
        return np.sqrt(2 / (n_ - 1)) * _gamma(n_ / 2) / _gamma((n_ - 1) / 2)

    if const_name == "c4":
        return _c4(n)
    if const_name == "d2":
        return 3.4699 - 6.8123 / n + 18.543 / n ** 2          # empirical fit
    if const_name == "d3":
        return 0.4573 + 0.0106 * n
    if const_name == "d4":
        return _CONST_TABLE[25][_COL_IDX["d4"]]               # nearly constant
    if const_name == "A2":
        d2_ = calc_unbiased_const("d2", n)
        return 3 / (d2_ * np.sqrt(n))
    if const_name == "A3":
        c4_ = _c4(n)
        return 3 / (c4_ * np.sqrt(n))
    if const_name in ("B3", "B4"):
        c4_ = _c4(n)
        k = 3 * np.sqrt(1 - c4_ ** 2) / c4_
        return max(0.0, 1 - k) if const_name == "B3" else 1 + k
    if const_name in ("D3", "D4"):
        d2_ = calc_unbiased_const("d2", n)
        d3_ = calc_unbiased_const("d3", n)
        k = 3 * d3_ / d2_
        return max(0.0, 1 - k) if const_name == "D3" else 1 + k

    raise ValueError(f"Approximation not implemented for '{const_name}' with n > 25.")


def calc_desc_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subgroup descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is a subgroup; each column is a measurement.
        The index is treated as the subgroup label.

    Returns
    -------
    pd.DataFrame
        Columns: subgroup, n, mean, std, range, min, max.
        Final row is the overall summary (labelled 'Overall').
    """
    numeric = df.select_dtypes(include="number")

    records = []
    for sg, row in numeric.iterrows():
        vals = row.dropna().values
        records.append({
            "subgroup": sg,
            "n":        len(vals),
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "range":    float(np.ptp(vals)),
            "min":      float(np.min(vals)),
            "max":      float(np.max(vals)),
        })

    all_vals = numeric.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    records.append({
        "subgroup": "Overall",
        "n":        len(all_vals),
        "mean":     float(np.mean(all_vals)),
        "std":      float(np.std(all_vals, ddof=1)),
        "range":    float(np.ptp(all_vals)),
        "min":      float(np.min(all_vals)),
        "max":      float(np.max(all_vals)),
    })

    return pd.DataFrame(records)


def shapiro_wilk_test(data: np.ndarray) -> dict:
    """
    Run the Shapiro-Wilk normality test.

    Parameters
    ----------
    data : array-like
        1-D sample (3 ≤ n ≤ 5000).

    Returns
    -------
    dict
        Keys: W (float), p_value (float), is_normal (bool).
        is_normal is True when p_value > 0.05.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n < 3:
        return {"W": np.nan, "p_value": np.nan, "is_normal": None,
                "note": "n < 3: Shapiro-Wilk not applicable."}

    W, p_value = stats.shapiro(arr)
    return {
        "W":        float(W),
        "p_value":  float(p_value),
        "is_normal": bool(p_value > 0.05),
    }


def calc_qq_data(data: np.ndarray) -> dict:
    """
    Compute Q-Q plot coordinates (sample vs theoretical normal quantiles).

    Parameters
    ----------
    data : array-like
        1-D sample.

    Returns
    -------
    dict
        Keys: theoretical (ndarray), sample (ndarray).
        Both arrays are the same length and ready to scatter-plot.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    (theoretical, sample), _ = stats.probplot(arr, dist="norm")
    return {
        "theoretical": np.array(theoretical, dtype=float),
        "sample":      np.array(sample,      dtype=float),
    }


def apply_boxcox(data: np.ndarray) -> dict:
    """
    Apply Box-Cox power transformation and report normality improvement.

    Parameters
    ----------
    data : array-like
        1-D positive sample (all values must be > 0).

    Returns
    -------
    dict
        Keys:
          transformed (ndarray) — Box-Cox transformed values,
          lambda      (float)   — optimal λ,
          p_before    (float)   — Shapiro-Wilk p-value before transform,
          p_after     (float)   — Shapiro-Wilk p-value after transform.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    if np.any(arr <= 0):
        raise ValueError("All values must be strictly positive for Box-Cox.")

    _, p_before = stats.shapiro(arr)

    transformed, lam = stats.boxcox(arr)

    _, p_after = stats.shapiro(transformed)

    return {
        "transformed": np.array(transformed, dtype=float),
        "lambda":      float(lam),
        "p_before":    float(p_before),
        "p_after":     float(p_after),
    }



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


import importlib.util
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

#03
# ── Local statistics.py (avoid shadowing stdlib statistics module) ───────────
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "statistics_local",
    os.path.join(_SELF_DIR, "statistics.py"),
)
assert _spec is not None and _spec.loader is not None, (
    "statistics.py를 찾을 수 없습니다. "
    "control_chart.py와 같은 폴더에 있는지 확인하세요."
)
_stats_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stats_mod)  # type: ignore[union-attr]
calc_unbiased_const = _stats_mod.calc_unbiased_const


# ── Internal helpers ─────────────────────────────────────────────────────────
def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def _broadcast(v: Any, n: int) -> np.ndarray:
    if np.isscalar(v):
        return np.full(n, float(v))
    return np.asarray(v, dtype=float)


def _make_chart_df(subgroups, points, cl, ucl, lcl) -> pd.DataFrame:
    n = len(points)
    return pd.DataFrame({
        "subgroup": list(subgroups),
        "point":    np.asarray(points, dtype=float),
        "CL":       _broadcast(cl,  n),
        "UCL":      _broadcast(ucl, n),
        "LCL":      _broadcast(lcl, n),
    })


def _get_count_and_size(
    df: pd.DataFrame,
    count_hints: Tuple[str, ...],
    size_hints: Tuple[str, ...],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Locate defect/defective count column and optional sample-size column.
    Tries named columns first, then falls back to positional (1st / 2nd numeric).
    """
    lower_map = {c.lower(): c for c in df.columns}
    num = _num(df)

    count_arr: Optional[np.ndarray] = None
    for h in count_hints:
        if h.lower() in lower_map:
            count_arr = df[lower_map[h.lower()]].values.astype(float)
            break
    if count_arr is None:
        if num.empty:
            raise ValueError("속성 관리도: 수치형 열을 찾을 수 없습니다.")
        count_arr = num.iloc[:,0].values.astype(float)

    size_arr: Optional[np.ndarray] = None
    for h in size_hints:
        if h.lower() in lower_map:
            size_arr = df[lower_map[h.lower()]].values.astype(float)
            break
    if size_arr is None and size_hints and num.shape[1] >= 2:
        size_arr = num.iloc[:,1].values.astype(float)

    return count_arr, size_arr


def _chart_summary(chart_df: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    if chart_df is None:
        return None
    return {
        "CL_mean":  float(chart_df["CL"].mean()),
        "UCL_mean": float(chart_df["UCL"].mean()),
        "LCL_mean": float(chart_df["LCL"].mean()),
    }


# ── Continuous chart builders ────────────────────────────────────────────────
def _xbar_r(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = _num(df)
    m = num.shape[1]
    if m < 2:
        raise ValueError("Xbar-R: 부분군 크기 ≥ 2 이어야 합니다.")

    subgroups = list(df.index)
    x_bars = num.mean(axis=1).values
    ranges  = (num.max(axis=1) - num.min(axis=1)).values

    x_bar_bar = float(np.mean(x_bars))
    r_bar     = float(np.mean(ranges))

    A2 = calc_unbiased_const("A2", m)
    D3 = calc_unbiased_const("D3", m)
    D4 = calc_unbiased_const("D4", m)

    upper_df = _make_chart_df(
        subgroups, x_bars,
        cl=x_bar_bar,
        ucl=x_bar_bar + A2 * r_bar,
        lcl=x_bar_bar - A2 * r_bar,
    )
    lower_df = _make_chart_df(
        subgroups, ranges,
        cl=r_bar,
        ucl=D4 * r_bar,
        lcl=max(0.0, D3 * r_bar),
    )
    return upper_df, lower_df


def _xbar_s(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = _num(df)
    m = num.shape[1]
    if m < 2:
        raise ValueError("Xbar-S: 부분군 크기 ≥ 2 이어야 합니다.")

    subgroups = list(df.index)
    x_bars = num.mean(axis=1).values
    s_vals  = num.std(axis=1, ddof=1).values

    x_bar_bar = float(np.mean(x_bars))
    s_bar     = float(np.mean(s_vals))

    A3 = calc_unbiased_const("A3", m)
    B3 = calc_unbiased_const("B3", m)
    B4 = calc_unbiased_const("B4", m)

    upper_df = _make_chart_df(
        subgroups, x_bars,
        cl=x_bar_bar,
        ucl=x_bar_bar + A3 * s_bar,
        lcl=x_bar_bar - A3 * s_bar,
    )
    lower_df = _make_chart_df(
        subgroups, s_vals,
        cl=s_bar,
        ucl=B4 * s_bar,
        lcl=max(0.0, B3 * s_bar),
    )
    return upper_df, lower_df


def _i_mr(df: pd.DataFrame, w: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = _num(df)
    if num.shape[1] > 1:
        raise ValueError(
            f"I-MR: 부분군 크기 = 1 이어야 합니다 (현재 {num.shape[1]}개 열). "
            "n > 1이면 Xbar-R 또는 Xbar-S를 사용하세요."
        )
    if not (2 <= w <= 5):
        raise ValueError("이동범위 윈도우 w는 2~5 사이여야 합니다.")

    subgroups = list(df.index)
    x = num.iloc[:, 0].values.astype(float)
    n = len(x)

    mr = np.array([
        x[max(0, i - w + 1):i + 1].max() - x[max(0, i - w + 1):i + 1].min()
        for i in range(n)
    ])
    # MR 평균은 완전한 윈도우(w개)가 확보된 구간부터 계산
    mr_bar = float(mr[w - 1:].mean()) if n >= w else float(mr.mean())

    x_bar = float(np.mean(x))
    d2    = calc_unbiased_const("d2", w)
    D3    = calc_unbiased_const("D3", w)
    D4    = calc_unbiased_const("D4", w)

    upper_df = _make_chart_df(
        subgroups, x,
        cl=x_bar,
        ucl=x_bar + 3.0 * mr_bar / d2,
        lcl=x_bar - 3.0 * mr_bar / d2,
    )
    lower_df = _make_chart_df(
        subgroups, mr,
        cl=mr_bar,
        ucl=D4 * mr_bar,
        lcl=max(0.0, D3 * mr_bar),
    )
    return upper_df, lower_df


# ── Attribute chart builders ─────────────────────────────────────────────────
def _np_chart(df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    subgroups = list(df.index)
    d, n_arr = _get_count_and_size(
        df,
        count_hints=("defectives", "d", "np"),
        size_hints=("n", "size", "sample_size"),
    )
    if n_arr is None:
        raise ValueError("NP 관리도: 샘플 크기 열('n' 또는 두 번째 수치 열)이 필요합니다.")

    n_fixed = float(np.mean(n_arr))
    np_bar  = float(np.mean(d))
    p_bar   = np_bar / n_fixed
    sigma   = float(np.sqrt(np_bar * (1.0 - p_bar)))

    upper_df = _make_chart_df(
        subgroups, d,
        cl=np_bar,
        ucl=np_bar + 3.0 * sigma,
        lcl=max(0.0, np_bar - 3.0 * sigma),
    )
    return upper_df, None


def _p_chart(df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    subgroups = list(df.index)
    d, n_arr = _get_count_and_size(
        df,
        count_hints=("defectives", "d", "p"),
        size_hints=("n", "n_i", "size"),
    )
    if n_arr is None:
        raise ValueError("P 관리도: 부분군 크기 열('n'/'n_i' 또는 두 번째 수치 열)이 필요합니다.")

    p_bar   = float(np.sum(d) / np.sum(n_arr))
    p_i     = d / n_arr
    sigma_i = np.sqrt(p_bar * (1.0 - p_bar) / n_arr)

    upper_df = _make_chart_df(
        subgroups, p_i,
        cl=p_bar,
        ucl=np.minimum(1.0, p_bar + 3.0 * sigma_i),
        lcl=np.maximum(0.0, p_bar - 3.0 * sigma_i),
    )
    return upper_df, None


def _c_chart(df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    subgroups = list(df.index)
    d, _ = _get_count_and_size(
        df,
        count_hints=("defects", "c", "count"),
        size_hints=(),
    )
    c_bar   = float(np.mean(d))
    sigma_c = float(np.sqrt(c_bar))

    upper_df = _make_chart_df(
        subgroups, d,
        cl=c_bar,
        ucl=c_bar + 3.0 * sigma_c,
        lcl=max(0.0, c_bar - 3.0 * sigma_c),
    )
    return upper_df, None


def _u_chart(df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
    subgroups = list(df.index)
    d, n_arr = _get_count_and_size(
        df,
        count_hints=("defects", "u", "count"),
        size_hints=("n", "n_i", "size"),
    )
    if n_arr is None:
        raise ValueError("U 관리도: 부분군 크기 열('n'/'n_i' 또는 두 번째 수치 열)이 필요합니다.")

    u_bar   = float(np.sum(d) / np.sum(n_arr))
    u_i     = d / n_arr
    sigma_i = np.sqrt(u_bar / n_arr)

    upper_df = _make_chart_df(
        subgroups, u_i,
        cl=u_bar,
        ucl=u_bar + 3.0 * sigma_i,
        lcl=np.maximum(0.0, u_bar - 3.0 * sigma_i),
    )
    return upper_df, None


# ── Public: calc_control_limits ──────────────────────────────────────────────
def calc_control_limits(
    df: pd.DataFrame,
    chart_type: str,
    mr_window: int = 2,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    관리도 통계량과 관리한계선을 계산합니다.

    Parameters
    ----------
    df : pd.DataFrame
        연속형 관리도 (Xbar-R/S, I-MR): 행=부분군, 수치형 열=측정값.
        속성 관리도 (NP/P/C/U): 첫 번째 수치형 열=불량/결점 수,
        두 번째 수치형 열 또는 'n'/'n_i' 열=부분군 크기.
    chart_type : str
        'Xbar-R' | 'Xbar-S' | 'I-MR' | 'NP' | 'P' | 'C' | 'U'
    mr_window : int, optional
        I-MR 이동범위 윈도우 크기 (2~5). 기본값 2.

    Returns
    -------
    upper_df : pd.DataFrame
        상단 관리도 (평균/개별값/속성). 열: subgroup, point, CL, UCL, LCL.
    lower_df : pd.DataFrame or None
        하단 관리도 (범위/표준편차/MR). 속성 관리도는 None.
    """
    key = chart_type.strip().upper().replace("-", "_").replace(" ", "_")
    dispatch: Dict[str, Any] = {
        "XBAR_R": _xbar_r,
        "XBAR_S": _xbar_s,
        "I_MR":   lambda d: _i_mr(d, mr_window),
        "IMR":    lambda d: _i_mr(d, mr_window),
        "NP":     _np_chart,
        "P":      _p_chart,
        "C":      _c_chart,
        "U":      _u_chart,
    }
    if key not in dispatch:
        valid = ["Xbar-R", "Xbar-S", "I-MR", "NP", "P", "C", "U"]
        raise ValueError(f"알 수 없는 관리도 유형: {chart_type!r}. 사용 가능: {valid}")
    return dispatch[key](df)


# ── Nelson's 8 Rules ──────────────────────────────────────────────────────────
def _sigma_per_point(chart_df: pd.DataFrame, sigma: Optional[float]) -> np.ndarray:
    """Per-point σ: use provided constant, or derive from (UCL - CL) / 3."""
    if sigma is not None:
        return np.full(len(chart_df), float(sigma))
    return (chart_df["UCL"].values - chart_df["CL"].values).astype(float) / 3.0


def apply_nelson_rules(
    chart_df: pd.DataFrame,
    sigma: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Nelson 8가지 규칙으로 이상 패턴을 검출합니다.

    Parameters
    ----------
    chart_df : pd.DataFrame
        calc_control_limits 출력값. 열: subgroup, point, CL, UCL, LCL.
    sigma : float, optional
        관리도 통계량의 표준편차. None이면 (UCL-CL)/3 으로 자동 계산.
        가변 한계선 관리도(P, U)는 None을 전달하면 점별 σ를 사용합니다.

    Returns
    -------
    list of dict
        각 항목: {rule_num, indices, subgroups, description}.
        위반이 없으면 빈 리스트.
    """
    pts  = chart_df["point"].values.astype(float)
    cl   = chart_df["CL"].values.astype(float)
    ucl  = chart_df["UCL"].values.astype(float)
    lcl  = chart_df["LCL"].values.astype(float)
    sgs  = list(chart_df["subgroup"])
    sig  = _sigma_per_point(chart_df, sigma)
    n    = len(pts)

    above   = pts > cl
    two_u   = cl + 2.0 * sig;  two_l = cl - 2.0 * sig
    one_u   = cl + sig;        one_l = cl - sig

    b2u     = pts > two_u;     b2l = pts < two_l
    b1u     = pts > one_u;     b1l = pts < one_l
    beyond1 = b1u | b1l
    within1 = (~b1u) & (~b1l)

    violations: List[Dict[str, Any]] = []

    def _record(rule_num: int, raw: List[int], desc: str) -> None:
        idx = sorted(set(raw))
        if idx:
            violations.append({
                "rule_num":    rule_num,
                "indices":     idx,
                "subgroups":   [sgs[i] for i in idx],
                "description": desc,
            })

    # Rule 1: 1점이 ±3σ(UCL/LCL) 이탈
    _record(1,
        [i for i in range(n) if pts[i] > ucl[i] or pts[i] < lcl[i]],
        "Rule 1: 관리한계선(±3σ) 이탈")

    # Rule 2: 중심선 한쪽에 연속 9점 이상
    r2: List[int] = []
    for i in range(8, n):
        w = above[i - 8:i + 1]
        if w.all() or (~w).all():
            r2.extend(range(i - 8, i + 1))
    _record(2, r2, "Rule 2: 중심선 한쪽에 연속 9점 이상")

    # Rule 3: 연속 6점 단조 증가 또는 단조 감소
    r3: List[int] = []
    for i in range(5, n):
        d = np.diff(pts[i - 5:i + 1])
        if (d > 0).all() or (d < 0).all():
            r3.extend(range(i - 5, i + 1))
    _record(3, r3, "Rule 3: 연속 6점 단조 증가 또는 단조 감소")

    # Rule 4: 연속 14점 교대 상승·하강
    r4: List[int] = []
    for i in range(13, n):
        d = np.diff(pts[i - 13:i + 1])
        if (d[:-1] * d[1:] < 0).all():
            r4.extend(range(i - 13, i + 1))
    _record(4, r4, "Rule 4: 연속 14점 교대 상승·하강")

    # Rule 5: 3점 중 2점이 ±2σ 바깥 (같은 쪽)
    r5: List[int] = []
    for i in range(2, n):
        if b2u[i - 2:i + 1].sum() >= 2:
            r5.extend(j for j in range(i - 2, i + 1) if b2u[j])
        if b2l[i - 2:i + 1].sum() >= 2:
            r5.extend(j for j in range(i - 2, i + 1) if b2l[j])
    _record(5, r5, "Rule 5: 3점 중 2점이 ±2σ 바깥 (같은 쪽)")

    # Rule 6: 5점 중 4점이 ±1σ 바깥 (같은 쪽)
    r6: List[int] = []
    for i in range(4, n):
        if b1u[i - 4:i + 1].sum() >= 4:
            r6.extend(j for j in range(i - 4, i + 1) if b1u[j])
        if b1l[i - 4:i + 1].sum() >= 4:
            r6.extend(j for j in range(i - 4, i + 1) if b1l[j])
    _record(6, r6, "Rule 6: 5점 중 4점이 ±1σ 바깥 (같은 쪽)")

    # Rule 7: 연속 15점이 ±1σ 이내 (층화 현상)
    r7: List[int] = []
    for i in range(14, n):
        if within1[i - 14:i + 1].all():
            r7.extend(range(i - 14, i + 1))
    _record(7, r7, "Rule 7: 연속 15점이 ±1σ 이내 (층화 현상)")

    # Rule 8: 연속 8점이 ±1σ 바깥 양쪽에 분포 (중심선 가까이 점 없음)
    r8: List[int] = []
    for i in range(7, n):
        wb = beyond1[i - 7:i + 1]
        if wb.all() and b1u[i - 7:i + 1].any() and b1l[i - 7:i + 1].any():
            r8.extend(range(i - 7, i + 1))
    _record(8, r8, "Rule 8: 연속 8점이 ±1σ 바깥 양쪽 분포")

    return violations


# ── Public: remove_ooc_recalculate ────────────────────────────────────────────
def remove_ooc_recalculate(
    df: pd.DataFrame,
    ooc_indices: List[int],
    chart_type: str,
    mr_window: int = 2,
) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.DataFrame]], Dict[str, Any]]:
    """
    OOC 부분군을 제거한 뒤 관리한계선을 재계산합니다.

    Parameters
    ----------
    df : pd.DataFrame
        calc_control_limits에 전달한 원본 데이터.
    ooc_indices : list of int
        제거할 행의 0-기반 위치 인덱스 (Nelson 규칙의 'indices' 값).
    chart_type : str
        calc_control_limits와 동일한 관리도 유형 문자열.
    mr_window : int, optional
        I-MR 윈도우 크기. 기본값 2.

    Returns
    -------
    cleaned_df : pd.DataFrame
        OOC 행이 제거된 DataFrame.
    new_limits : tuple (upper_df, lower_df)
        cleaned_df로 재계산된 관리한계선.
    comparison : dict
        제거 전후 비교.
        키: removed_indices, removed_subgroups,
            upper_before, upper_after, lower_before, lower_after.
        각 차트 요약: {CL_mean, UCL_mean, LCL_mean}.
    """
    upper_before, lower_before = calc_control_limits(df, chart_type, mr_window)

    if not ooc_indices:
        return df.copy(), (upper_before, lower_before), {
            "removed_indices":   [],
            "removed_subgroups": [],
            "upper_before":      _chart_summary(upper_before),
            "upper_after":       _chart_summary(upper_before),
            "lower_before":      _chart_summary(lower_before),
            "lower_after":       _chart_summary(lower_before),
        }

    valid = sorted({i for i in ooc_indices if 0 <= i < len(df)})
    removed_subgroups = list(df.iloc[valid].index)

    keep = np.ones(len(df), dtype=bool)
    for i in valid:
        keep[i] = False
    cleaned_df = df.iloc[keep].copy()

    if cleaned_df.empty:
        raise ValueError("모든 부분군이 OOC로 제거됩니다 — 재계산 불가.")

    upper_after, lower_after = calc_control_limits(cleaned_df, chart_type, mr_window)

    comparison: Dict[str, Any] = {
        "removed_indices":   valid,
        "removed_subgroups": removed_subgroups,
        "upper_before":      _chart_summary(upper_before),
        "upper_after":       _chart_summary(upper_after),
        "lower_before":      _chart_summary(lower_before),
        "lower_after":       _chart_summary(lower_after),
    }
    return cleaned_df, (upper_after, lower_after), comparison


import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#04
# ── Chart title maps ─────────────────────────────────────────────────────────
_UPPER_TITLES: Dict[str, str] = {
    "XBAR_R": "X̄ 관리도 (평균)",
    "XBAR_S": "X̄ 관리도 (평균)",
    "I_MR":   "I 관리도 (개별값)",
    "IMR":    "I 관리도 (개별값)",
    "NP":     "NP 관리도 (불량 개수)",
    "P":      "P 관리도 (불량률)",
    "C":      "C 관리도 (결점 수)",
    "U":      "U 관리도 (단위당 결점)",
}
_LOWER_TITLES: Dict[str, str] = {
    "XBAR_R": "R 관리도 (범위)",
    "XBAR_S": "S 관리도 (표준편차)",
    "I_MR":   "MR 관리도 (이동범위)",
    "IMR":    "MR 관리도 (이동범위)",
}

# ── Shared style constants ───────────────────────────────────────────────────
_COL_DATA  = "#3b82f6"   # blue  – measurements
_COL_CL    = "#22c55e"   # green – center line
_COL_UCL   = "#ec4899"   # magenta – UCL
_COL_LCL   = "#ef4444"   # red – LCL
_COL_OOC   = "#dc2626"   # red – OOC markers
_COL_LSL   = "#ef4444"   # red – spec limit
_COL_TGT   = "#3b82f6"   # blue – target line


# ── Private helpers ───────────────────────────────────────────────────────────
def _chart_key(chart_type: str) -> str:
    return chart_type.strip().upper().replace("-", "_").replace(" ", "_")


def _upper_title(chart_type: str) -> str:
    return _UPPER_TITLES.get(_chart_key(chart_type), f"{chart_type} 관리도")


def _lower_title(chart_type: str) -> str:
    return _LOWER_TITLES.get(_chart_key(chart_type), "하단 관리도")


def _fmt(name: str, val: Optional[float]) -> str:
    """Format a capability index value for annotation text."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"<b>{name}</b> = N/A"
    return f"<b>{name}</b> = {val:.4f}"


def _is_const(arr: np.ndarray) -> bool:
    """True when all values in arr are (nearly) equal."""
    return bool(np.allclose(arr, arr[0], atol=1e-10, rtol=1e-8))


def _add_chart_traces(
    fig: go.Figure,
    chart_df: pd.DataFrame,
    ooc_idx: List[int],
    row: int,
    show_legend: bool,
) -> None:
    """Add data line, CL/UCL/LCL lines, and OOC markers to one subplot row."""
    x    = list(chart_df["subgroup"])
    pts  = chart_df["point"].values.astype(float)
    cl   = chart_df["CL"].values.astype(float)
    ucl  = chart_df["UCL"].values.astype(float)
    lcl  = chart_df["LCL"].values.astype(float)

    # ── Data line ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=pts,
        mode="lines+markers",
        name="측정값",
        line=dict(color=_COL_DATA, width=1.5),
        marker=dict(color=_COL_DATA, size=5),
        legendgroup="data",
        showlegend=show_legend,
    ), row=row, col=1)

    # ── OOC points ─────────────────────────────────────────────────────────
    valid_ooc = [i for i in ooc_idx if 0 <= i < len(x)]
    if valid_ooc:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in valid_ooc],
            y=[pts[i] for i in valid_ooc],
            mode="markers",
            name="OOC (이상점)",
            marker=dict(color=_COL_OOC, size=12, symbol="circle-open", line=dict(width=2)),
            legendgroup="ooc",
            showlegend=show_legend,
        ), row=row, col=1)

    # ── CL ─────────────────────────────────────────────────────────────────
    cl_const = _is_const(cl)
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]] if cl_const else x,
        y=[cl[0], cl[0]] if cl_const else cl,
        mode="lines",
        name=f"CL = {cl[0]:.4f}" if cl_const else "CL",
        line=dict(color=_COL_CL, dash="dash", width=1.5),
        legendgroup="cl",
        showlegend=show_legend,
    ), row=row, col=1)

    # ── UCL ────────────────────────────────────────────────────────────────
    ucl_const = _is_const(ucl)
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]] if ucl_const else x,
        y=[ucl[0], ucl[0]] if ucl_const else ucl,
        mode="lines",
        name=f"UCL = {ucl[0]:.4f}" if ucl_const else "UCL",
        line=dict(color=_COL_UCL, dash="dot", width=1.5),
        legendgroup="ucl",
        showlegend=show_legend,
    ), row=row, col=1)

    # ── LCL ────────────────────────────────────────────────────────────────
    lcl_const = _is_const(lcl)
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]] if lcl_const else x,
        y=[lcl[0], lcl[0]] if lcl_const else lcl,
        mode="lines",
        name=f"LCL = {lcl[0]:.4f}" if lcl_const else "LCL",
        line=dict(color=_COL_LCL, dash="dot", width=1.5),
        legendgroup="lcl",
        showlegend=show_legend,
    ), row=row, col=1)


# ── Public functions ──────────────────────────────────────────────────────────
def plot_capability_chart(
    data: Any,
    USL: Optional[float],
    LSL: Optional[float],
    target: Optional[float],
    indices: Dict[str, Optional[float]],
) -> go.Figure:
    """
    히스토그램 + 정규분포 곡선 + 규격선 + 공정능력지수 주석 박스.

    Parameters
    ----------
    data    : array-like  개별 측정값 전체 (1-D로 자동 flatten)
    USL     : float|None  상한 규격
    LSL     : float|None  하한 규격
    target  : float|None  목표값
    indices : dict        Cp, Cpk, Pp, Ppk 값 (None 허용)

    Returns
    -------
    plotly Figure
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]

    mu    = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))

    # x 범위: 규격선을 포함할 수 있도록 양쪽 여유
    x_lo = min(arr.min(), LSL if LSL is not None else arr.min()) - 2.5 * sigma
    x_hi = max(arr.max(), USL if USL is not None else arr.max()) + 2.5 * sigma
    x_pdf = np.linspace(x_lo, x_hi, 400)
    pdf   = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / sigma) ** 2)

    fig = go.Figure()

    # Histogram (확률 밀도로 정규화 → PDF와 같은 y 스케일)
    fig.add_trace(go.Histogram(
        x=arr,
        histnorm="probability density",
        name="측정값 분포",
        marker=dict(
            color="rgba(59,130,246,0.45)",
            line=dict(color="rgba(59,130,246,0.8)", width=1),
        ),
        hovertemplate="구간: %{x}<br>밀도: %{y:.4f}<extra></extra>",
    ))

    # Normal PDF curve
    fig.add_trace(go.Scatter(
        x=x_pdf, y=pdf,
        mode="lines",
        name="정규분포 곡선",
        line=dict(color="#1e40af", width=2.5),
        hovertemplate="x: %{x:.4f}<br>밀도: %{y:.4f}<extra></extra>",
    ))

    # ±3σ shaded zone (within spec visual guide)
    three_lo = mu - 3 * sigma
    three_hi = mu + 3 * sigma
    fig.add_vrect(
        x0=three_lo, x1=three_hi,
        fillcolor="rgba(34,197,94,0.06)",
        layer="below",
        line_width=0,
    )

    # LSL / USL 수직 점선
    for val, label, side in [
        (LSL,    "LSL", "top left"),
        (USL,    "USL", "top right"),
    ]:
        if val is not None:
            fig.add_vline(
                x=val,
                line=dict(color=_COL_LSL, dash="dash", width=2),
                annotation=dict(
                    text=f"<b>{label}</b>={val:.4f}",
                    font=dict(color=_COL_LSL, size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                ),
                annotation_position=side,
            )

    # Target 수직 점선
    if target is not None:
        fig.add_vline(
            x=target,
            line=dict(color=_COL_TGT, dash="dot", width=2),
            annotation=dict(
                text=f"<b>Target</b>={target:.4f}",
                font=dict(color=_COL_TGT, size=11),
                bgcolor="rgba(255,255,255,0.8)",
            ),
            annotation_position="top",
        )

    # 공정능력지수 주석 박스 (우측 상단)
    idx_lines = [
        _fmt("Cp",  indices.get("Cp")),
        _fmt("Cpk", indices.get("Cpk")),
        _fmt("Pp",  indices.get("Pp")),
        _fmt("Ppk", indices.get("Ppk")),
    ]
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        text="<br>".join(idx_lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#9ca3af",
        borderwidth=1,
        borderpad=6,
        font=dict(family="monospace", size=12),
        xanchor="right",
        yanchor="top",
    )

    fig.update_layout(
        title=dict(text="공정능력 분포도", font=dict(size=16)),
        xaxis_title="측정값",
        yaxis_title="확률 밀도",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        bargap=0.05,
    )
    return fig


def plot_qq(qq_data: Dict[str, Any]) -> go.Figure:
    """
    Q-Q Plot: 이론 정규분위수 vs 표본 분위수.

    Parameters
    ----------
    qq_data : dict  {'theoretical': ndarray, 'sample': ndarray}
              calc_qq_data() 반환값과 동일한 구조.

    Returns
    -------
    plotly Figure
    """
    theoretical = np.asarray(qq_data["theoretical"], dtype=float)
    sample      = np.asarray(qq_data["sample"],      dtype=float)

    # 기준선 y=x 의 범위
    lo = float(min(theoretical.min(), sample.min()))
    hi = float(max(theoretical.max(), sample.max()))
    margin = (hi - lo) * 0.05
    ref_x = [lo - margin, hi + margin]

    fig = go.Figure()

    # 기준선 y=x
    fig.add_trace(go.Scatter(
        x=ref_x, y=ref_x,
        mode="lines",
        name="기준선 (y = x)",
        line=dict(color="red", dash="dash", width=1.5),
    ))

    # 표본 분위수 산점도
    fig.add_trace(go.Scatter(
        x=theoretical, y=sample,
        mode="markers",
        name="표본 분위수",
        marker=dict(color=_COL_DATA, size=7, opacity=0.8,
                    line=dict(color="#1e40af", width=0.5)),
        hovertemplate="이론: %{x:.4f}<br>표본: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Q-Q Plot (정규성 검정)", font=dict(size=16)),
        xaxis_title="이론 분위수 (표준 정규)",
        yaxis_title="표본 분위수",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_control_chart(
    upper_df: pd.DataFrame,
    lower_df: Optional[pd.DataFrame],
    ooc_indices: List[int],
    chart_type: str,
) -> go.Figure:
    """
    상단(평균/개별값/속성) + 하단(범위/σ/MR) 관리도를 2행 서브플롯으로 표시.

    Parameters
    ----------
    upper_df    : DataFrame  calc_control_limits() 반환 상단 차트 데이터
    lower_df    : DataFrame|None  하단 차트 데이터; 속성 관리도는 None
    ooc_indices : list of int  상단 차트의 OOC 행 위치 (0-based)
    chart_type  : str  관리도 유형 문자열

    Returns
    -------
    plotly Figure
    """
    has_lower = lower_df is not None
    n_rows    = 2 if has_lower else 1
    sub_titles = (
        [_upper_title(chart_type), _lower_title(chart_type)]
        if has_lower else
        [_upper_title(chart_type)]
    )

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=sub_titles,
        row_heights=[0.55, 0.45] if has_lower else [1.0],
    )

    _add_chart_traces(fig, upper_df, ooc_indices, row=1, show_legend=True)

    if has_lower:
        _add_chart_traces(fig, lower_df, [], row=2, show_legend=False)

    fig.update_layout(
        title=dict(text=f"{chart_type} 관리도", font=dict(size=16)),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=1.01, y=1,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d1d5db",
            borderwidth=1,
        ),
        height=520 if has_lower else 320,
    )

    # x축 제목은 가장 아래 행에만
    fig.update_xaxes(title_text="부분군", row=n_rows, col=1)

    return fig


def plot_boxplot(
    df: pd.DataFrame,
    USL: Optional[float] = None,
    LSL: Optional[float] = None,
) -> go.Figure:
    """
    부분군별 상자 그림 (Box Plot).

    Parameters
    ----------
    df  : DataFrame  행=부분군, 수치형 열=측정값 (calc_desc_stats 입력과 동일 구조)
    USL : float|None
    LSL : float|None

    Returns
    -------
    plotly Figure
    """
    numeric = df.select_dtypes(include="number")

    fig = go.Figure()

    palette = [
        "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
        "#f59e0b", "#f97316", "#ec4899", "#6366f1",
    ]

    for k, sg in enumerate(numeric.index):
        vals = numeric.loc[sg].dropna().values.astype(float)
        color = palette[k % len(palette)]
        fig.add_trace(go.Box(
            y=vals,
            name=str(sg),
            boxpoints="all",
            jitter=0.35,
            pointpos=-1.6,
            marker=dict(color=color, size=5, opacity=0.7),
            line=dict(color=color),
            fillcolor=f"rgba({int(color[1:3], 16)},"
                      f"{int(color[3:5], 16)},"
                      f"{int(color[5:7], 16)},0.3)",
            hovertemplate=f"<b>{sg}</b><br>%{{y:.4f}}<extra></extra>",
        ))

    # LSL / USL 수평 기준선
    for val, label in [(LSL, "LSL"), (USL, "USL")]:
        if val is not None:
            fig.add_hline(
                y=val,
                line=dict(color=_COL_LSL, dash="dash", width=2),
                annotation=dict(
                    text=f"<b>{label}</b> = {val:.4f}",
                    font=dict(color=_COL_LSL, size=11),
                    bgcolor="rgba(255,255,255,0.8)",
                    xanchor="right",
                ),
                annotation_position="right",
            )

    fig.update_layout(
        title=dict(text="부분군별 상자 그림 (Box Plot)", font=dict(size=16)),
        xaxis_title="부분군",
        yaxis_title="측정값",
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_run_chart(
    data: Any,
    x_bar: float,
) -> go.Figure:
    """
    런 차트: 개별 측정값을 시간 순서로 표시 + 전체 평균 기준선.

    Parameters
    ----------
    data  : array-like  측정값 전체 (2-D이면 행 순서로 flatten)
    x_bar : float       전체 평균 (수평 기준선)

    Returns
    -------
    plotly Figure
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n   = len(arr)
    idx = list(range(1, n + 1))

    fig = go.Figure()

    # 측정값 선 + 마커
    fig.add_trace(go.Scatter(
        x=idx, y=arr,
        mode="lines+markers",
        name="개별 측정값",
        line=dict(color=_COL_DATA, width=1.2),
        marker=dict(color=_COL_DATA, size=4, opacity=0.7),
        hovertemplate="순서: %{x}<br>값: %{y:.4f}<extra></extra>",
    ))

    # 전체 평균 기준선
    fig.add_hline(
        y=x_bar,
        line=dict(color=_COL_CL, dash="dash", width=2),
        annotation=dict(
            text=f"<b>x̄</b> = {x_bar:.4f}",
            font=dict(color=_COL_CL, size=12),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        annotation_position="right",
    )

    fig.update_layout(
        title=dict(text="런 차트 (Run Chart)", font=dict(size=16)),
        xaxis_title="측정 순서",
        yaxis_title="측정값",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


import importlib.util
import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# ── Module loading ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(modname: str, fname: str):
    for base in [os.path.join(_ROOT, "modules"), _ROOT]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(modname, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[modname] = mod
                return mod
    raise FileNotFoundError(
        f"'{fname}'을 찾을 수 없습니다 — modules/ 또는 {_ROOT} 확인"
    )


_s  = _load("spc_statistics",    "statistics.py")
_c  = _load("spc_capability",    "capability.py")
_cc = _load("spc_control_chart", "control_chart.py")
_v  = _load("spc_visualization", "visualization.py")

calc_desc_stats         = _s.calc_desc_stats
shapiro_wilk_test       = _s.shapiro_wilk_test
calc_qq_data            = _s.calc_qq_data
apply_boxcox            = _s.apply_boxcox
calc_sigma_within       = _c.calc_sigma_within
calc_sigma_overall      = _c.calc_sigma_overall
calc_capability_indices = _c.calc_capability_indices
calc_control_limits     = _cc.calc_control_limits
apply_nelson_rules      = _cc.apply_nelson_rules
remove_ooc_recalculate  = _cc.remove_ooc_recalculate
plot_capability_chart   = _v.plot_capability_chart
plot_qq                 = _v.plot_qq
plot_control_chart      = _v.plot_control_chart
plot_boxplot            = _v.plot_boxplot
plot_run_chart          = _v.plot_run_chart

# ── Sample data ───────────────────────────────────────────────────────────────
_SAMPLE_DF = pd.DataFrame(
    {
        "pl_1": [3576.27, 3504.17, 3440.11, 3638.33, 3661.94],
        "pl_2": [3630.12, 3514.52, 3494.35, 3719.84, 3485.53],
        "pl_3": [3576.27, 3747.43, 3962.93, 3617.47, 3499.43],
        "pl_4": [3630.12, 3666.15, 3514.30, 3450.17, 3605.53],
        "pl_5": [3355.69, 3709.25, 3273.57, 3378.70, 3390.29],
        "pl_6": [3363.62, 3317.28, 3336.20, 3475.50, 3519.26],
    },
    index=["m1", "m2", "m3", "m4", "m5"],
).T
_SAMPLE_DF.index.name = "Lot"
_SAMPLE_DF.columns = [f"m{i+1}" for i in range(5)]

_CONTINUOUS = {"Xbar-R", "Xbar-S", "I-MR"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cpk_bg(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#6b7280"
    if val >= 1.67: return "#16a34a"
    if val >= 1.33: return "#2563eb"
    if val >= 1.00: return "#d97706"
    if val >= 0.67: return "#ea580c"
    return "#dc2626"


# 수정 코드
def _metric_card(label: str, val, sub: str = "") -> str:
    is_na  = val is None or (isinstance(val, float) and np.isnan(float(val)))
    vstr   = "N/A" if is_na else f"{float(val):.4f}"
    color  = _cpk_bg(None if is_na else float(val))
    return (
        f'<div style="background:{color};color:white;'
        f'padding:12px 6px;'
        f'border-radius:12px;text-align:center;'
        f'min-height:90px;'
        f'display:flex;flex-direction:column;'
        f'justify-content:center;align-items:center;'
        f'box-sizing:border-box;width:100%;overflow:hidden">'
        f'<div style="font-size:.85rem;font-weight:600;'
        f'opacity:.9;white-space:nowrap">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;'
        f'margin:4px 0;white-space:nowrap">{vstr}</div>'
        f'<div style="font-size:.68rem;opacity:.8;'
        f'white-space:nowrap">{sub}</div>'
        f'</div>'
    )


def _fv(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return "N/A" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def _load_df(src) -> pd.DataFrame:
    buf = io.StringIO(src) if isinstance(src, str) else src
    df  = pd.read_csv(buf)
    df  = df.set_index(df.columns[0])
    return df.select_dtypes(include="number")


def _run_analysis(df, USL, LSL, target, chart_type, mr_window) -> dict:
    if df is None or df.empty:
        raise ValueError("데이터가 비어있습니다.")

    num      = df.select_dtypes(include="number")
    all_vals = num.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)].astype(float)

    if len(all_vals) == 0:
        raise ValueError("유효한 숫자 데이터가 없습니다.")

    try:
        desc = calc_desc_stats(df)
    except Exception as e:
        raise RuntimeError(f"[calc_desc_stats 실패] {e}")

    try:
        shapiro = shapiro_wilk_test(all_vals)
    except Exception as e:
        raise RuntimeError(f"[shapiro_wilk_test 실패] {e}")

    try:
        qq_data = calc_qq_data(all_vals)
    except Exception as e:
        raise RuntimeError(f"[calc_qq_data 실패] {e}")

    x_bar   = float(np.mean(all_vals))
    is_cont = chart_type in _CONTINUOUS

    if is_cont:
        try:
            swr = calc_sigma_within(df)
        except Exception as e:
            raise RuntimeError(f"[calc_sigma_within 실패] {e}")

        try:
            sor = calc_sigma_overall(df)
        except Exception as e:
            raise RuntimeError(f"[calc_sigma_overall 실패] {e}")

        try:
            cap = calc_capability_indices(
                x_bar         = x_bar,
                sigma_within  = swr["sigma_within"],
                sigma_overall = sor["sigma_overall"],
                USL           = USL,
                LSL           = LSL,
            )
        except Exception as e:
            raise RuntimeError(f"[calc_capability_indices 실패] {e}")
    else:
        swr = sor = cap = None

    try:
        upper_df, lower_df = calc_control_limits(df, chart_type, mr_window)
    except Exception as e:
        raise RuntimeError(f"[calc_control_limits 실패] {e}")

    try:
        nelson = apply_nelson_rules(upper_df)
    except Exception as e:
        raise RuntimeError(f"[apply_nelson_rules 실패] {e}")

    ooc_idx = sorted({i for v in nelson for i in v["indices"]})

    return dict(
        desc_stats           = desc,
        all_vals             = all_vals,
        x_bar                = x_bar,
        shapiro              = shapiro,
        qq_data              = qq_data,
        sigma_within_result  = swr,
        sigma_overall_result = sor,
        capability           = cap,
        upper_df             = upper_df,
        lower_df             = lower_df,
        nelson               = nelson,
        ooc_idx              = ooc_idx,
        USL                  = USL,
        LSL                  = LSL,
        target               = target,
        chart_type           = chart_type,
        mr_window            = mr_window,
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="공정능력분석 & SPC 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {
    padding-top: 3rem !important;        /* 1.2rem → 3rem 으로 증가 */
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: .9rem;
    font-weight: 600;
}
[data-testid="column"] {
    padding: 0 4px !important;
}
h1, h2, h3 {
    word-break: keep-all !important;
    white-space: normal !important;
    line-height: 1.5 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
_DEFAULTS = dict(
    df=None, results=None, ooc_results=None,
    USL=4000.0, LSL=3000.0, target=3500.0,
    use_usl=True, use_lsl=True, use_target=True,
    chart_type="Xbar-R", mr_window=2,
)
for _k, _dv in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _dv


# ═══════════════════════════════════════════════════════
# SIDEBAR  ← 버튼 포함 전체를 with 블록 안에
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ SPC 분석 설정")

    # 데이터 입력
    st.subheader("📂 데이터 입력")

    if st.button("📦 샘플 데이터 로드 (PVC 점도)",
                 use_container_width=True):
        st.session_state.df          = _SAMPLE_DF.copy()
        st.session_state.USL         = 4000.0
        st.session_state.LSL         = 3000.0
        st.session_state.target      = 3500.0
        st.session_state.results     = None
        st.session_state.ooc_results = None
        st.success("샘플 데이터 로드 완료")

    uploaded = st.file_uploader("또는 CSV 파일 업로드", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.df          = _load_df(uploaded)
            st.session_state.results     = None
            st.session_state.ooc_results = None
            st.success(f"✅ {len(st.session_state.df)}개 부분군 로드 완료")
        except Exception as e:
            st.error(f"파일 오류: {e}")

    if st.session_state.df is not None:
        _df = st.session_state.df
        st.info(f"📋 부분군 **{len(_df)}개** × 측정값 **{_df.shape[1]}개**")

    st.divider()

    # 규격 설정
    st.subheader("📏 규격 설정")
    for _key, _label, _default in [
        ("USL",    "USL (상한 규격)", 4000.0),
        ("LSL",    "LSL (하한 규격)", 3000.0),
        ("target", "Target (목표값)", 3500.0),
    ]:
        _use_key = f"use_{_key.lower()}"
        _c1, _c2 = st.columns([1, 3])
        with _c1:
            st.session_state[_use_key] = st.checkbox(
                _key,
                value=st.session_state[_use_key],
                key=f"cb_{_key}",
            )
        with _c2:
            st.session_state[_key] = st.number_input(
                _label,
                value=float(st.session_state[_key]),
                disabled=not st.session_state[_use_key],
                label_visibility="collapsed",
                key=f"ni_{_key}",
            )

    st.divider()

    # 관리도 설정
    st.subheader("📈 관리도 설정")
    _force_imr = (
        st.session_state.df is not None
        and st.session_state.df.select_dtypes("number").shape[1] == 1
    )
    _ct_opts = ["I-MR"] if _force_imr else [
        "Xbar-R", "Xbar-S", "I-MR", "NP", "P", "C", "U"
    ]
    if _force_imr:
        st.caption("⚠️ 부분군 크기=1 → I-MR 전용")

    st.session_state.chart_type = st.selectbox(
        "관리도 유형", _ct_opts, key="sel_chart"
    )
    if st.session_state.chart_type == "I-MR":
        st.session_state.mr_window = st.select_slider(
            "이동범위 윈도우 (w)", [2, 3, 4, 5],
            value=st.session_state.mr_window,
            key="sl_mrw",
        )

    st.divider()

    # ── 분석 실행 버튼 (with st.sidebar 안에 있어야 함) ──
    if st.button("🔍 분석 실행", type="primary",
                 use_container_width=True):
        if st.session_state.df is None:
            st.error("먼저 데이터를 로드하세요.")
        else:
            with st.spinner("분석 중…"):
                try:
                    st.session_state.results = _run_analysis(
                        st.session_state.df,
                        USL        = st.session_state.USL    if st.session_state.use_usl    else None,
                        LSL        = st.session_state.LSL    if st.session_state.use_lsl    else None,
                        target     = st.session_state.target if st.session_state.use_target else None,
                        chart_type = st.session_state.chart_type,
                        mr_window  = st.session_state.mr_window,
                    )
                    st.session_state.ooc_results = None
                    st.success("✅ 분석 완료!")
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    st.session_state.results = None
                except Exception as e:
                    st.error(f"❌ 예기치 못한 오류: {e}")
                    st.error(traceback.format_exc())
                    st.session_state.results = None

# ═══════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════

def render_dashboard():
    """results가 있을 때만 호출되는 함수"""
    R = st.session_state.results

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 공정능력분석",
        "📈 관리도",
        "📉 Q-Q Plot",
        "📦 Box & Run",
        "📋 요약 리포트",
    ])

    # ── TAB 1 ──────────────────────────────────────────
    with tab1:
        cap     = R.get("capability")
        is_cont = R["chart_type"] in _CONTINUOUS

        if not is_cont:
            st.info("ℹ️ 속성 관리도(NP/P/C/U)에서는 공정능력지수를 계산하지 않습니다.")
        else:
            st.subheader("공정능력지수")
            _tiles = [
                ("Cp",  "단기 정밀도"),
                ("Cpk", "단기 치우침 반영"),
                ("Pp",  "장기 정밀도"),
                ("Ppk", "장기 치우침 반영"),
            ]
            for _col, (_name, _sub) in zip(st.columns(4), _tiles):
                _val = cap.get(_name) if cap else None
                _col.markdown(
                    _metric_card(_name, _val if _val is not None else float("nan"), _sub),
                    unsafe_allow_html=True,
                )
            st.caption("")

            if cap:
                _gl = cap.get("grade_label", "")
                _ac = cap.get("action", "")
                if _gl:
                    _color = _cpk_bg(cap.get("Cpk"))
                    st.markdown(
                        f'<div style="background:{_color}22;border-left:4px solid {_color};'
                        f'padding:10px 14px;border-radius:6px;margin:8px 0">'
                        f'<b>판정:</b> {_gl} &nbsp;|&nbsp; <b>권장 조치:</b> {_ac}</div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("📊 등급 기준표", expanded=False):
                st.dataframe(
                    pd.DataFrame([
                        ("≥ 1.67",      "등급 0 (특급)",   "현 공정 유지, 관리도 주기 완화 검토"),
                        ("1.33 ~ 1.67", "등급 1 (우수)",   "현 공정 유지, 지속 모니터링"),
                        ("1.00 ~ 1.33", "등급 2 (보통)",   "공정 개선 검토, 변동 원인 분석"),
                        ("0.67 ~ 1.00", "등급 3 (불량)",   "즉시 원인 조사 및 시정 조치"),
                        ("< 0.67",      "등급 4 (부적합)", "생산 중단, 공정 전면 재검토"),
                    ], columns=["Cpk 범위", "등급", "권장 조치"]),
                    use_container_width=True, hide_index=True,
                )

            st.divider()

            st.subheader("정규성 검정 (Shapiro-Wilk)")
            sw = R["shapiro"]
            _sc1, _sc2, _sc3 = st.columns(3)
            _sc1.metric("W 통계량", _fv(sw.get("W")))
            _sc2.metric("p-값",     _fv(sw.get("p_value")))
            with _sc3:
                if sw.get("is_normal") is True:
                    st.success("✅ PASS — 정규 분포 (α=0.05)")
                elif sw.get("is_normal") is False:
                    st.warning("⚠️ FAIL — 비정규 분포 (Box-Cox 권장)")
                else:
                    st.info(sw.get("note", "N/A"))

            st.divider()

            swr = R.get("sigma_within_result")  or {}
            sor = R.get("sigma_overall_result") or {}
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("σ_within",        _fv(swr.get("sigma_within")))
            _mc2.metric("σ_overall",       _fv(sor.get("sigma_overall")))
            _mc3.metric("전체 평균 (x̄̄)", _fv(R["x_bar"]))
            st.divider()

        st.subheader("공정능력 분포도")
        _idx4chart = {k: (cap.get(k) if cap else None) for k in ["Cp","Cpk","Pp","Ppk"]}
        st.plotly_chart(
            plot_capability_chart(R["all_vals"], R["USL"], R["LSL"], R["target"], _idx4chart),
            use_container_width=True,
        )

        with st.expander("📊 부분군별 기술 통계량", expanded=False):
            _desc     = R["desc_stats"].copy()
            _num_cols = [c for c in _desc.columns if c not in ("subgroup", "n")]
            st.dataframe(
                _desc.style.format({c: "{:.4f}" for c in _num_cols}),
                use_container_width=True, hide_index=True,
            )

    # ── TAB 2 ──────────────────────────────────────────
    with tab2:
        _ooc_r = st.session_state.ooc_results
        if _ooc_r:
            _upper   = _ooc_r["upper_df"]
            _lower   = _ooc_r["lower_df"]
            _nelson  = _ooc_r["nelson"]
            _ooc_pts = _ooc_r["ooc_idx"]
            st.info(f"✅ OOC 부분군 {len(_ooc_r['comparison']['removed_indices'])}개 제거 후 재계산")
        else:
            _upper   = R["upper_df"]
            _lower   = R["lower_df"]
            _nelson  = R["nelson"]
            _ooc_pts = R["ooc_idx"]

        st.plotly_chart(
            plot_control_chart(_upper, _lower, _ooc_pts, R["chart_type"]),
            use_container_width=True,
        )

        st.subheader("Nelson 규칙 위반 현황")
        if not _nelson:
            st.success("✅ 모든 Nelson 규칙을 만족합니다.")
        else:
            st.error(f"⚠️ {len(_nelson)}개 이상 패턴 감지")
            st.dataframe(
                pd.DataFrame([
                    {
                        "규칙":       f"Rule {v['rule_num']}",
                        "설명":       v["description"],
                        "이상 점 수": len(v["indices"]),
                        "부분군":     ", ".join(str(s) for s in v["subgroups"]),
                    }
                    for v in _nelson
                ]),
                use_container_width=True, hide_index=True,
            )

            if _ooc_r is None:
                if st.button("🗑️ OOC 부분군 제거 후 재계산", type="secondary"):
                    with st.spinner("재계산 중…"):
                        try:
                            _cdf, (_nu, _nl), _cmp = remove_ooc_recalculate(
                                st.session_state.df,
                                R["ooc_idx"],
                                R["chart_type"],
                                R["mr_window"],
                            )
                            _new_nel = apply_nelson_rules(_nu)
                            st.session_state.ooc_results = dict(
                                upper_df   = _nu,
                                lower_df   = _nl,
                                nelson     = _new_nel,
                                ooc_idx    = sorted({i for v in _new_nel for i in v["indices"]}),
                                comparison = _cmp,
                                cleaned_df = _cdf,
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"재계산 오류: {e}")
            else:
                with st.expander("📊 제거 전후 관리한계선 비교", expanded=True):
                    _cmp  = _ooc_r["comparison"]
                    _rows = []
                    for _lbl, _bk, _ak in [
                        ("상단 관리도", "upper_before", "upper_after"),
                        ("하단 관리도", "lower_before", "lower_after"),
                    ]:
                        _b, _a = _cmp.get(_bk), _cmp.get(_ak)
                        if _b and _a:
                            _rows.append({
                                "차트":     _lbl,
                                "CL (전)":  f"{_b['CL_mean']:.4f}",
                                "UCL (전)": f"{_b['UCL_mean']:.4f}",
                                "LCL (전)": f"{_b['LCL_mean']:.4f}",
                                "CL (후)":  f"{_a['CL_mean']:.4f}",
                                "UCL (후)": f"{_a['UCL_mean']:.4f}",
                                "LCL (후)": f"{_a['LCL_mean']:.4f}",
                            })
                    if _rows:
                        st.dataframe(pd.DataFrame(_rows),
                                     use_container_width=True, hide_index=True)

                if st.button("↩️ 원래 데이터로 복원"):
                    st.session_state.ooc_results = None
                    st.rerun()

    # ── TAB 3 ──────────────────────────────────────────
    with tab3:
        _apply_bc = st.checkbox("🔄 Box-Cox 변환 적용", help="양수 데이터에서만 가능")
        _bc = None
        if _apply_bc:
            try:
                _bc = apply_boxcox(R["all_vals"])
            except ValueError as e:
                st.error(f"Box-Cox 적용 불가: {e}")

        _qc1, _qc2 = st.columns([1, 2])
        with _qc1:
            st.subheader("정규성 검정")
            sw = R["shapiro"]
            st.metric("Shapiro-Wilk W", _fv(sw.get("W")))
            st.metric("p-값",           _fv(sw.get("p_value")))
            if sw.get("is_normal") is True:
                st.success("PASS — 정규 분포")
            elif sw.get("is_normal") is False:
                st.warning("FAIL — 비정규 분포")
            else:
                st.info(sw.get("note", "N/A"))

            if _bc is not None:
                st.divider()
                st.subheader("Box-Cox 결과")
                st.metric("최적 λ",      f"{_bc['lambda']:.4f}")
                st.metric("p (변환 전)", f"{_bc['p_before']:.4f}")
                st.metric("p (변환 후)", f"{_bc['p_after']:.4f}")
                if _bc["p_after"] > _bc["p_before"]:
                    st.success("정규성 개선됨")
                else:
                    st.warning("변환 효과 미미")

        with _qc2:
            st.subheader("Q-Q Plot (원본)")
            st.plotly_chart(plot_qq(R["qq_data"]), use_container_width=True)
            if _bc is not None:
                st.subheader("Q-Q Plot (Box-Cox 변환 후)")
                st.plotly_chart(
                    plot_qq(calc_qq_data(_bc["transformed"])),
                    use_container_width=True,
                )

    # ── TAB 4 ──────────────────────────────────────────
    with tab4:
        _bc1, _bc2 = st.columns(2)
        with _bc1:
            st.subheader("부분군별 Box Plot")
            st.plotly_chart(
                plot_boxplot(st.session_state.df, R["USL"], R["LSL"]),
                use_container_width=True,
            )
        with _bc2:
            st.subheader("런 차트 (Run Chart)")
            st.plotly_chart(
                plot_run_chart(R["all_vals"], R["x_bar"]),
                use_container_width=True,
            )

    # ── TAB 5 ──────────────────────────────────────────
    with tab5:
        st.subheader("📋 분석 결과 요약")
        _cap = R.get("capability") or {}
        sw   = R["shapiro"]
        swr  = R.get("sigma_within_result")  or {}
        sor  = R.get("sigma_overall_result") or {}

        _summary_rows = [
            ("분석 정보", "관리도 유형",      R["chart_type"]),
            ("분석 정보", "부분군 수",        str(len(st.session_state.df))),
            ("분석 정보", "측정값 수/부분군", str(st.session_state.df.shape[1])),
            ("분석 정보", "총 측정값",        str(len(R["all_vals"]))),
            ("규격",     "USL",             _fv(R["USL"])),
            ("규격",     "LSL",             _fv(R["LSL"])),
            ("규격",     "Target",          _fv(R["target"])),
            ("기술 통계", "전체 평균 (x̄̄)", _fv(R["x_bar"])),
            ("기술 통계", "σ_within",        _fv(swr.get("sigma_within"))),
            ("기술 통계", "σ_overall",       _fv(sor.get("sigma_overall"))),
            ("공정능력",  "Cp",              _fv(_cap.get("Cp"))),
            ("공정능력",  "Cpk",             _fv(_cap.get("Cpk"))),
            ("공정능력",  "Pp",              _fv(_cap.get("Pp"))),
            ("공정능력",  "Ppk",             _fv(_cap.get("Ppk"))),
            ("공정능력",  "등급 (Cpk 기준)", _cap.get("grade_label", "N/A")),
            ("공정능력",  "권장 조치",        _cap.get("action", "N/A")),
            ("정규성",   "Shapiro-Wilk W",  _fv(sw.get("W"))),
            ("정규성",   "p-값",             _fv(sw.get("p_value"))),
            ("정규성",   "판정 (α=0.05)",   "PASS" if sw.get("is_normal") else "FAIL"),
            ("관리도",   "Nelson 위반 수",   str(len(R["nelson"]))),
            ("관리도",   "OOC 점 수",        str(len(R["ooc_idx"]))),
        ]
        _sum_df = pd.DataFrame(_summary_rows, columns=["카테고리", "항목", "값"])
        st.dataframe(_sum_df, use_container_width=True, hide_index=True)
        st.divider()

        _dc1, _dc2, _dc3 = st.columns(3)
        with _dc1:
            st.download_button(
                "⬇️ 요약 결과 CSV",
                _sum_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="spc_summary.csv", mime="text/csv",
                use_container_width=True,
            )
        with _dc2:
            st.download_button(
                "⬇️ 기술 통계 CSV",
                R["desc_stats"].to_csv(index=False, encoding="utf-8-sig"),
                file_name="desc_stats.csv", mime="text/csv",
                use_container_width=True,
            )
        with _dc3:
            st.download_button(
                "⬇️ 관리도 데이터 CSV",
                R["upper_df"].to_csv(index=False, encoding="utf-8-sig"),
                file_name="control_chart.csv", mime="text/csv",
                use_container_width=True,
            )

        if R["nelson"]:
            st.divider()
            st.subheader("Nelson 위반 상세 목록")
            _vd = []
            for _v in R["nelson"]:
                for _idx, _sg in zip(_v["indices"], _v["subgroups"]):
                    _vd.append({
                        "규칙":    f"Rule {_v['rule_num']}",
                        "설명":    _v["description"],
                        "행 위치": _idx,
                        "부분군":  str(_sg),
                    })
            st.dataframe(pd.DataFrame(_vd), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# 진입점 — if/else 로 분기 (st.stop 사용 안 함)
# ═══════════════════════════════════════════════════════
if st.session_state.get("results") is None:
    st.markdown(
        """
        <div style="padding-top:1rem;margin-bottom:1rem">
            <p style="
                font-size:1.4rem;
                font-weight:700;
                line-height:1.8;
                word-break:keep-all;
                white-space:normal;
                margin:0;
            ">
            📊 공정능력분석 &amp; 통계적공정관리 대시보드
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("← 사이드바에서 데이터를 로드하고 **분석 실행** 버튼을 클릭하세요.")
    st.subheader("📋 샘플 데이터 미리보기 (PVC 점도)")
    st.dataframe(_SAMPLE_DF, use_container_width=True)
    st.caption("USL = 4000 | LSL = 3000 | Target = 3500")
else:
    render_dashboard()


