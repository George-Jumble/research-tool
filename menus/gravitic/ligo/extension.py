"""LIGO GWOSC strain fetch + ringdown analysis module."""

from __future__ import annotations

from typing import Any

from menus.gravitic.ligo.fetcher import fetch_and_graph as pull_strain

MODULE_TAG = "LIGO_GWOSC"
SUBMENU_TITLE = "LIGO GWOSC (Strain Data)"

REPORT_ACTION = "Full Ringdown Report"
SCAN_ACTION = "Scan Post-Merger Markers"

MENU_ACTIONS = [
    "Pull Strain from GWOSC",
    REPORT_ACTION,
    SCAN_ACTION,
]


def entry_fields(action: str) -> list[dict]:
    from sb_shared.llm_analysis import append_llm_entry_fields

    if action == "Pull Strain from GWOSC":
        return append_llm_entry_fields(
            [
                {
                    "key": "query",
                    "label": "Event / segment / URI",
                    "default": "GW150914",
                    "required": True,
                    "hint": "GW150914, HEAVY_BBH (10-event preset), GW150914,GW190521,…, H:H1_R:start:end, or osdf:///…",
                },
                {
                    "key": "force_refresh",
                    "label": "Force refresh",
                    "default": "no",
                    "required": False,
                    "hint": "yes = re-download even if cached",
                },
            ]
        )
    return append_llm_entry_fields(
        [
            {
                "key": "event",
                "label": "GW event ID",
                "default": "ALL",
                "required": False,
                "hint": "ALL / * / blank = every cached event; or one ID (e.g. GW150914)",
            },
            {
                "key": "strain_path",
                "label": "Strain file (optional)",
                "default": "",
                "required": False,
                "hint": "Specific cached .hdf5/.txt.gz; blank = all detectors",
            },
            {
                "key": "detector_mode",
                "label": "Detector mode",
                "default": "coherent",
                "required": False,
                "hint": "coherent = H1+L1+V1 stack + null-stream veto | per_detector (strain_path forces it)",
            },
            {
                "key": "postmerger_scan",
                "label": "Post-merger scan",
                "default": "yes",
                "required": False,
                "hint": "yes = 20–300 ms echo/secondary-marker scan with time-slide FAP (coherent mode)",
            },
            {
                "key": "show_graphics",
                "label": "Graphics mode",
                "default": "artifacts",
                "required": False,
                "hint": "artifacts = PNG only | popup = PNG + window | no = no plot",
            },
        ]
    )


def entry_instructions(action: str) -> list[str]:
    if action == "Pull Strain from GWOSC":
        return [
            "Fetches open strain via GWOSC Event API, GWDataFind, or Pelican/OSDF.",
            "Caches under datasets/gwosc/ with resume ledger done.txt.",
            "Requires: gwdatafind, requests-pelican, pelican CLI (install_gravitic_deps.sh).",
        ]
    return [
        "Operates on cached GWOSC strain (.hdf5 or .txt.gz).",
        "Scored by the Superblock residual engine (template E).",
        "Report: artifacts/superblock_residuals/LIGO_GWOSC_*.json",
        "Plot: artifacts/ligo_gwosc/ligo_ringdown_residuals.png (f and γ vs GR Kerr, pulls, post-merger scan);",
        "  Scan Post-Merger Markers also writes artifacts/ligo_gwosc/ligo_postmerger_scan.png.",
        "Blind damped-sinusoid fit (f, τ) on the coherent H1+L1+V1 stack; residual rows f and γ = 1/(2πτ) in Hz.",
        "Spin from events/<EVENT>/event_detail.json (auto-fetched). Batch-pull: query HEAVY_BBH.",
        "Run Pull Strain first if datasets/gwosc/events/<EVENT>/ is empty.",
    ]


PLOT_NAME = "ligo_ringdown_residuals.png"
SCAN_PLOT_NAME = "ligo_postmerger_scan.png"
_NO_PLOT = {"no", "n", "none", "off", "false", "0"}


def _graphics_mode(options: dict[str, Any]) -> str:
    from sb_shared.run_output import parse_show_graphics

    raw = str(options.get("show_graphics", "artifacts") or "artifacts").strip().lower()
    if raw in _NO_PLOT:
        return "no"
    return "popup" if parse_show_graphics(options, default="artifacts") else "artifacts"


def _out_path(options: dict, default: str) -> Any:
    from sb_shared.project_paths import SUPERBLOCK_ROOT

    out_dir = SUPERBLOCK_ROOT / "artifacts" / "ligo_gwosc"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(options.get("output_name") or default).strip()
    if not name.endswith(".png"):
        name += ".png"
    return out_dir / name


def _finish(fig, out, show_popup: bool):
    import matplotlib.pyplot as plt

    fig.savefig(out, dpi=150)
    print(f"[Superblock Residual Engine] Plot saved: {out}")
    if show_popup:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def _plot_scan_panel(ax, points: list[dict]) -> None:
    scans = [(p["label"], p["postmerger_scan"]) for p in points if p.get("postmerger_scan")]
    if not scans:
        ax.text(0.5, 0.5, "post-merger scan not run", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    idx = list(range(len(scans)))
    sig = [s["sigma_equiv"] for _, s in scans]
    colors = ["tab:red" if s["detected"] else ("tab:gray" if not s["null_veto_passed"] else "tab:blue") for _, s in scans]
    ax.bar(idx, sig, color=colors)
    ax.axhline(3.0, color="tab:red", ls="--", lw=0.8, label="3σ detection threshold")
    for i, (_, s) in enumerate(scans):
        ax.annotate(f"{s['delay_ms']:.0f} ms\n{s['f_hz']:.0f} Hz", (i, max(s["sigma_equiv"], 0)), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6)
    ax.set_xticks(idx, [l for l, _ in scans], rotation=35, ha="right", fontsize=7)
    ax.set_ylim(min(-1.0, min(sig)) - 0.2, max(3.8, max(sig) + 0.8))
    ax.set_ylabel("σ_equiv (time-slide FAP)")
    ax.set_title("Post-merger secondary markers, 20–300 ms", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def _plot_residual_report(report: dict, options: dict, show_popup: bool) -> str | None:
    points = [p for p in (report.get("points") or [])]
    if not points:
        print(f"[Superblock Residual Engine] LIGO plot skipped: no residuals ({report.get('load_error') or 'N=0'})")
        return None
    import matplotlib.pyplot as plt

    out = _out_path(options, PLOT_NAME)
    labels = [p["label"] for p in points]
    idx = list(range(len(points)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_f, ax_g), (ax_p, ax_s) = axes
    ax_f.errorbar(idx, [p["f_obs_hz"] for p in points], yerr=[p["sigma_meas_hz"] for p in points], fmt="o", capsize=3,
                  label="observed f (blind fit, σ_fit)")
    ax_f.errorbar(idx, [p["f220_gr_hz"] for p in points], yerr=[p["sigma_gr_hz"] for p in points], fmt="s", mfc="none",
                  capsize=3, label="GR Kerr f_220 (σ_GR)")
    ax_f.set_ylabel("f [Hz]")
    ax_f.set_title("Oscillation frequency", fontsize=9)
    have_g = [i for i, p in enumerate(points) if "gamma_obs_hz" in p]
    ax_g.errorbar(have_g, [points[i]["gamma_obs_hz"] for i in have_g],
                  yerr=[points[i]["sigma_gamma_meas_hz"] for i in have_g], fmt="o", capsize=3,
                  label="observed γ = 1/(2πτ)")
    ax_g.errorbar(idx, [p["gamma220_gr_hz"] for p in points], yerr=[p["sigma_gamma_gr_hz"] for p in points], fmt="s",
                  mfc="none", capsize=3, label="GR γ_220 = f_220/(2Q_220)")
    for i in have_g:
        p = points[i]
        ax_g.annotate(f"τ {p['tau_ms']:.1f}/{p['tau220_gr_ms']:.1f} ms", (i, p["gamma_obs_hz"]), textcoords="offset points",
                      xytext=(6, -10), fontsize=6)
    ax_g.set_ylabel("γ [Hz]")
    ax_g.set_title("Damping rate (τ obs/GR annotated)", fontsize=9)
    for ax in (ax_f, ax_g):
        ax.set_xticks(idx, labels, rotation=35, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(points) - 0.4)
        ax.legend(fontsize=7, loc="best")
    ax_p.axhspan(-1, 1, color="0.9", label="±1σ")
    ax_p.axhline(0.0, color="k", lw=0.8)
    ax_p.errorbar([i - 0.12 for i in idx], [p["pull"] for p in points], yerr=[1.0] * len(points), fmt="o", capsize=3,
                  label="f: (D − B)/σ")
    gi = [i for i in idx if points[i].get("pull_gamma") is not None]
    ax_p.errorbar([i + 0.12 for i in gi], [points[i]["pull_gamma"] for i in gi], yerr=[1.0] * len(gi), fmt="D", capsize=3,
                  label="γ: (D − B)/σ")
    for i, p in enumerate(points):
        badge = "VETO" if p.get("vetoed") else ("null ✓" if p.get("null_veto_passed") and p.get("null_stream_chi2_red") is not None else "1 det")
        ax_p.annotate(f"SNR {p.get('network_snr', p['snr_peak']):.1f}\n{badge}", (i, ax_p.get_ylim()[0]),
                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)
    ax_p.set_xticks(idx, labels, rotation=35, ha="right", fontsize=7)
    ax_p.set_xlim(-0.6, len(points) - 0.4)
    ax_p.set_ylabel("pull [σ]")
    ax_p.set_title("Standardised pulls (network SNR, null-stream badge)", fontsize=9)
    ax_p.legend(fontsize=7, loc="upper right")
    _plot_scan_panel(ax_s, points)
    chi2 = report.get("chi2_Lambda")
    chi2_txt = f"{chi2:.2f}" if isinstance(chi2, (int, float)) else "n/a"
    j = report.get("joint_constraints") or {}
    jf, jg = j.get("delta_f_over_f") or {}, j.get("delta_gamma_over_gamma") or {}
    jt = ""
    if jf:
        jt += f"  ⟨δf/f⟩={jf['mean']:+.3f}±{jf['sigma']:.3f}"
    if jg:
        jt += f"  ⟨δγ/γ⟩={jg['mean']:+.3f}±{jg['sigma']:.3f}"
    fig.suptitle(f"LIGO ringdown vs GR Kerr (2,2,0)  χ²_Λ={chi2_txt}  N={report.get('N')}{jt}", fontsize=11)
    fig.tight_layout()
    return _finish(fig, out, show_popup)


def _plot_postmerger(report: dict, options: dict, show_popup: bool) -> str | None:
    points = report.get("points") or []
    if not any(p.get("postmerger_scan") for p in points):
        print("[Superblock Residual Engine] Post-merger plot skipped: no scan results")
        return None
    import matplotlib.pyplot as plt

    out = _out_path({"output_name": options.get("scan_output_name")}, SCAN_PLOT_NAME)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    _plot_scan_panel(ax1, points)
    scans = [(p["label"], p["postmerger_scan"]) for p in points if p.get("postmerger_scan")]
    idx = list(range(len(scans)))
    ax2.errorbar(idx, [s["network_snr"] for _, s in scans], fmt="o", label="loudest gated network ρ")
    ax2.plot(idx, [s["background_median"] for _, s in scans], "k_", ms=14, label="background median")
    ax2.plot(idx, [s["background_p99"] for _, s in scans], "r_", ms=14, label="background 99th pct")
    ax2.set_xticks(idx, [l for l, _ in scans], rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("max-over-bank ρ")
    ax2.legend(fontsize=7)
    ax2.set_title("Candidate vs off-source/time-slide background", fontsize=9)
    fig.suptitle("Post-merger damped-sinusoid scan (f 50–400 Hz, τ 4/10/25 ms, delay 20–300 ms)", fontsize=11)
    fig.tight_layout()
    return _finish(fig, out, show_popup)


def run_action(action: str, options: dict[str, Any] | None = None) -> str | None:
    from superblock_mode import CONTINUE, begin_live_action

    options = dict(options or {})
    if action == SCAN_ACTION:
        options["postmerger_scan"] = "yes"
        options.setdefault("detector_mode", "coherent")
    routed = begin_live_action(MODULE_TAG, action, options)
    if routed is not CONTINUE:
        if isinstance(routed, dict):
            mode = _graphics_mode(options)
            if action in (REPORT_ACTION, SCAN_ACTION) and mode != "no":
                routed["plot_path"] = _plot_residual_report(routed, options, mode == "popup" and action == REPORT_ACTION)
            if action == SCAN_ACTION and mode != "no":
                routed["scan_plot_path"] = _plot_postmerger(routed, options, mode == "popup")
            return routed.get("report_path")
        return routed
    options = options or {}
    if action == "Pull Strain from GWOSC":
        query = str(options.get("query", "")).strip()
        if not query:
            print("[LIGO GWOSC] Missing query/event ID.")
            return
        pull_strain(query, options)
        return

    print(f"[LIGO GWOSC] Unknown action: {action}")
    return None
