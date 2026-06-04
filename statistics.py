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
