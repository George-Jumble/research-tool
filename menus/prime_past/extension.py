"""
Prime Past residuals ↔ research_tool.py

AlterBBN v2.2 is the standard-BBN baseline (author ruling 2026-11-04).

  Fetch & Build AlterBBN v2.2   loader: pinned upstream tarball → datasets/prime_past/, compile both
                                drivers, cache the Planck baseline CSV, write bbn_isolated/gates.json + report.json
  BBN Gate Check (AlterBBN)     loader: G1–G5 from the AlterBBN output (bbn_isolated/runner.py)
  Score BBN Residuals           residual engine (template D): D = Cooke 2018 D/H, Aver 2021 Y_p; B = AlterBBN
  Plot BBN Residuals            same, plus pulls plot under artifacts/prime_past/ (pop-up when Graphics = popup)
  Full BBN Residual Report      same as Plot

Old action names (Verification Suite, Generate Visualizations, Export LaTeX Appendix, Monte-Carlo Domain
Scan) did nothing beyond an N=0 note; they still resolve (see LEGACY_ACTIONS) but are not on the menu.
"""

from __future__ import annotations

from pathlib import Path

from sb_shared.run_output import parse_show_graphics

MODULE_TAG = "PRIME_PAST_HARMONIC"
SUBMENU_TITLE = "Prime Past Residuals"

FETCH_ACTION = "Fetch & Build AlterBBN v2.2"
GATE_ACTION = "BBN Gate Check (AlterBBN)"
SCORE_ACTION = "Score BBN Residuals"
PLOT_ACTION = "Plot BBN Residuals"
REPORT_ACTION = "Full BBN Residual Report"

MENU_ACTIONS = [FETCH_ACTION, GATE_ACTION, SCORE_ACTION, PLOT_ACTION, REPORT_ACTION]
LOADER_ACTIONS = frozenset({FETCH_ACTION, GATE_ACTION})

_ACTION_CONFIG = {
    SCORE_ACTION: {"plot": False},
    PLOT_ACTION: {"plot": True},
    REPORT_ACTION: {"plot": True},
}

LEGACY_ACTIONS = {
    "Verification Suite": SCORE_ACTION,
    "Generate Visualizations": PLOT_ACTION,
    "Export LaTeX Appendix": REPORT_ACTION,
    "Monte-Carlo Domain Scan": SCORE_ACTION,
}


def is_module_selection(repo: str | None) -> bool:
    return repo == MODULE_TAG


def _yes(value) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def entry_fields(action: str) -> list[dict]:
    if action == FETCH_ACTION:
        return [{
            "key": "force_refresh", "label": "Force rebuild", "default": "no", "required": False,
            "hint": "yes = re-verify the tarball and recompile even if binaries exist", "choices": ["no", "yes"],
        }]
    if action == GATE_ACTION:
        return []
    return [
        {"key": "output_name", "label": "Plot filename (optional)", "default": "prime_past_bbn_residuals.png",
         "required": False, "hint": "Saved under artifacts/prime_past/ for plot actions"},
        {"key": "show_graphics", "label": "Graphics mode", "default": "artifacts", "required": False,
         "hint": "popup = Tk window | artifacts = PNG only", "choices": ["artifacts", "popup"]},
    ]


def entry_instructions(action: str) -> list[str]:
    if action == FETCH_ACTION:
        return [
            "Fetches AlterBBN v2.2 (hepforge, sha256-pinned) into datasets/prime_past/ and compiles",
            "sb_bbn.x + sb_bbn_full.x with cc/gcc/clang (no OpenMP). Skipped if binaries already exist.",
            "Then caches the Planck-input baseline and writes bbn_isolated/gates.json + report.json.",
            "SB_ALTERBBN_DIR = use an existing tree; SB_ALTERBBN_NO_AUTO_BUILD=1 = never fetch/build.",
        ]
    if action == GATE_ACTION:
        return [
            "Runs AlterBBN at eta=6.114e-10, tau_n=877.75 s, N_eff=3.044 and computes gates G1–G5.",
            "Writes bbn_isolated/gates.json and report.json.",
        ]
    return [
        "Scored by the Superblock residual engine (template D; SB prediction pending, no formula).",
        "D = Cooke 2018 D/H and Aver 2021 Y_p; B = AlterBBN v2.2 at Planck inputs (only if G1–G5 pass).",
        "AlterBBN is fetched/built automatically on first use when online and a C compiler exists.",
        "Report: artifacts/superblock_residuals/PRIME_PAST_HARMONIC_*.json; plots under artifacts/prime_past/.",
    ]


def fetch_and_build(options: dict | None = None, *, verbose: bool = True) -> dict:
    """Fetch + build AlterBBN, cache the Planck baseline and refresh gates.json / report.json."""
    from bbn_isolated import alterbbn
    from bbn_isolated.runner import run as run_gates

    opts = options or {}
    info = alterbbn.ensure_alterbbn(force=_yes(opts.get("force_refresh")), verbose=verbose)
    base = alterbbn.run()
    gates = run_gates(check_isolated=False)
    if verbose:
        v = base["values"]
        print(f"[Superblock Residual Engine] AlterBBN baseline: Y_p={v['Y_p']:.5f} D/H={v['D_H']:.4e} "
              f"(cache {Path(base['cache_path']).name if base.get('cache_path') else 'none'})")
    return {"alterbbn": info, "baseline": base, "gates": gates}


def gate_check(*, verbose: bool = True) -> dict:
    from bbn_isolated.runner import HERE, run as run_gates

    payload = run_gates(check_isolated=False)
    if verbose:
        print(f"[Superblock Residual Engine] Gate report: {HERE / 'report.json'}")
    return payload


def _plot_residual_report(report: dict, options: dict, show_popup: bool) -> str | None:
    table = report.get("table") or []
    if not table:
        print(f"[Superblock Residual Engine] Prime Past plot skipped: no residuals "
              f"({report.get('load_error') or 'N=0'})")
        return None
    import matplotlib.pyplot as plt

    from sb_shared.project_paths import SUPERBLOCK_ROOT

    out_dir = SUPERBLOCK_ROOT / "artifacts" / "prime_past"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (options.get("output_name") or "prime_past_bbn_residuals.png").strip()
    if not name.endswith(".png"):
        name += ".png"
    out = out_dir / name
    labels = [row["observable"] for row in table]
    pulls = [row["pull"] for row in table]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(-1, 1, color="0.9", label="±1σ (obs)")
    ax.axhline(0.0, color="k", lw=0.8, label="AlterBBN baseline")
    ax.errorbar(range(len(table)), pulls, yerr=[1.0] * len(table), fmt="o", ms=6, capsize=3,
                label="(D − B)/σ")
    for i, row in enumerate(table):
        ax.annotate(f"D={row['D']:.4g}\nB={row['B']:.4g}", (i, row["pull"]), textcoords="offset points",
                    xytext=(10, -10), fontsize=8)
    ax.set_xticks(range(len(table)), labels)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylabel("pull [σ]")
    ax.set_title(f"Prime Past BBN residuals (Template D pending)  χ²_Λ={report.get('chi2_Lambda'):.2f}  "
                 f"N={report.get('N')}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[Superblock Residual Engine] Plot saved: {out}")
    if show_popup:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def run_action(
    selection: str,
    show_plots: bool = True,
    options: dict | None = None,
) -> str | None:
    """Execute a Prime Past action (loader actions here; scoring through the residual engine)."""
    from superblock_mode import CONTINUE, begin_live_action

    options = options or {}
    selection = LEGACY_ACTIONS.get(selection, selection)
    if selection == FETCH_ACTION:
        from bbn_isolated import alterbbn

        try:
            fetch_and_build(options)
        except alterbbn.AlterBBNUnavailable as exc:
            print(f"[Superblock Residual Engine] AlterBBN not available: {exc}")
            return None
        print(f"[Superblock Residual Engine] Next: {SCORE_ACTION} or {REPORT_ACTION}.")
        from bbn_isolated.runner import HERE

        return str(HERE / "report.json")
    if selection == GATE_ACTION:
        payload = gate_check()
        from bbn_isolated.runner import HERE

        return str(HERE / "report.json") if payload else None
    routed = begin_live_action(MODULE_TAG, selection, options)
    if routed is CONTINUE:
        print(f"[Superblock Residual Engine] Unknown Prime Past action: {selection}")
        return None
    config = _ACTION_CONFIG.get(selection)
    if isinstance(routed, dict) and config and config["plot"]:
        show_popup = parse_show_graphics(options, default="artifacts") if show_plots else False
        routed["plot_path"] = _plot_residual_report(routed, options, show_popup)
    if isinstance(routed, dict):
        return routed.get("report_path")
    return routed
