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
