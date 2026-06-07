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
