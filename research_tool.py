#!/usr/bin/env python3
"""
Superblock Research Tool — thin entry point.

Each submenu lives in its own ``*_extension.py`` module.  Shared curses UI,
menu tree, and run orchestration live under ``sb_research/``.

Run:  ./venv/bin/python research_tool.py

Remote AI: configure keys in config/api_keys.env (see api_keys.env.example).
Menu domain **Remote AI Processing** — test connections, analyze artifacts, custom prompts.
Post-run review: set **AI verbose review = yes** on any module entry form.
"""

from __future__ import annotations

# sb_research is not imported at module load. The curses UI imports it only
# when this file is executed with no engine subcommand.

_ENGINE_COMMANDS = (
    "geometry",
    "sparc",
    "desi",
    "cms",
    "casimir",
    "bounce",
    "corpus",
    "neutron",
)


def __getattr__(name: str):
    """Lazy curses re-exports. Importing this module does not load sb_research."""
    from sb_research import curses_shell, n_selector, registry

    table = {
        "ROOT_TITLE": curses_shell.ROOT_TITLE,
        "ChunkedScanProgress": curses_shell.ChunkedScanProgress,
        "_safe_addstr": curses_shell._safe_addstr,
        "append_nav": curses_shell.append_nav,
        "draw_menu": curses_shell.draw_menu,
        "draw_progress_bar": curses_shell.draw_progress_bar,
        "select_n_interactive": curses_shell.select_n_interactive,
        "show_entry_form": curses_shell.show_entry_form,
        "should_offer_n_selector": n_selector.should_offer_n_selector,
        "MODULE_EXTENSIONS": registry.MODULE_EXTENSIONS,
        "module_entry_fields": registry.module_entry_fields,
        "module_entry_instructions": registry.module_entry_instructions,
    }
    if name in table:
        return table[name]
    raise AttributeError(name)


def _missing_engine(command: str) -> int:
    from pathlib import Path

    root = Path(__file__).resolve().parent
    constants = root / "sb_engine" / "constants.py"
    if command == "geometry" and constants.is_file():
        from sb_engine.constants import (
            CORPUS_DATE,
            DELTA_TEP_MEV,
            HBAR_C_MEV_FM,
            R_M_LOCKED_FM,
            R_M_LOCKED_M,
        )

        print(f"[Superblock Residual Engine] CORPUS_DATE={CORPUS_DATE}")
        print(f"[Superblock Residual Engine] DELTA_TEP_MEV={DELTA_TEP_MEV}")
        print(f"[Superblock Residual Engine] r_M = hbar*c/DELTA_TEP = {R_M_LOCKED_FM} fm")
        print(f"[Superblock Residual Engine] R_M_LOCKED_M={R_M_LOCKED_M}")
        print(f"[Superblock Residual Engine] HBAR_C_MEV_FM={HBAR_C_MEV_FM}")
        return 0
    searched = [
        constants,
        root / "sb_engine" / f"{command}.py",
        root / f"{command}.py",
    ]
    print(f"[Superblock Residual Engine] `{command}` is not in this tree.")
    for path in searched:
        print(f"  searched {path} exists={path.is_file()}")
    return 0


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] in _ENGINE_COMMANDS:
        raise SystemExit(_missing_engine(argv[0]))
    from sb_research.runner import load_api_keys, main

    load_api_keys()
    main()