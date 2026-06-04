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
