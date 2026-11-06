"""
Curses main loop — Primary Matrix navigation and action dispatch.

Submenu-specific logic is delegated to each ``*_extension.py`` via
``handle_pre_form``, ``entry_fields``, and ``entry_instructions``.
"""

from __future__ import annotations

import curses

from sb_shared.llm_analysis import append_llm_entry_fields
from sb_shared.clear_cache import CLEAR_CACHE_MENU_LABEL
from sb_shared.clear_cache_menu import run_clear_cache_menu
from sb_shared.test_set_reset import (
    RESET_MENU_LABEL,
    confirm_reset,
    reset_test_set,
    resolve_module_tag,
)

from sb_research.curses_shell import (
    NAV_BACK,
    NAV_HOME,
    ROOT_TITLE,
    _safe_addstr,
    append_nav,
    draw_menu,
    show_entry_form,
)
from sb_research.data_pull import data_pull_entry_fields, data_pull_entry_instructions
from sb_research.dataset_manager import (
    DATASET_FETCH_ACTION,
    DATASET_MANAGER_TAG,
    DATASET_RUN_ACTION,
)
from sb_research.external_editor import (
    EXTERNAL_EDITOR_MENU_LABEL,
    load_external_editor_command,
    save_external_editor_command,
)
from sb_research.gateway_launcher import (
    GATEWAY_GUI_MENU_LABEL,
    gateway_log_path,
    is_gateway_running,
    spawn_gateway_gui,
)
from sb_research.menu_tree import build_menu_tree
from sb_research.n_selector import seed_field_defaults_from_params, strip_auto_filled_fields
from sb_research.registry import (
    handle_module_pre_form,
    module_entry_fields,
    module_entry_instructions,
)

def main_curses(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

    domains, repos, submenus, module_action_map = build_menu_tree()
    root_menu = append_nav(domains)

    menu_history = []
    current_row = 0
    current_menu = root_menu
    menu_title = ROOT_TITLE

    def go_home():
        nonlocal current_menu, menu_title, current_row, menu_history
        menu_history.clear()
        current_menu = root_menu
        menu_title = ROOT_TITLE
        current_row = 0

    def go_back():
        nonlocal current_menu, menu_title, current_row
        if menu_history:
            menu_title, current_menu, current_row = menu_history.pop()
        else:
            go_home()

    def descend_into(selection):
        nonlocal current_menu, menu_title, current_row, menu_history
        menu_history.append((menu_title, current_menu, current_row))
        menu_title = selection
        current_menu = append_nav(repos[selection])
        current_row = 0

    def descend_into_submenu(selection):
        nonlocal current_menu, menu_title, current_row, menu_history
        menu_history.append((menu_title, current_menu, current_row))
        menu_title = selection
        current_menu = append_nav(submenus[selection])
        current_row = 0

    while True:
        breadcrumb = " > ".join(entry[0] for entry in menu_history + [(menu_title,)])
        draw_menu(stdscr, menu_title, current_menu, current_row, breadcrumb=breadcrumb)
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(current_menu) - 1:
            current_row += 1
        elif key in (10, 13):
            selection = current_menu[current_row]

            if selection == "Exit":
                return None, None, {}
            if selection == NAV_HOME:
                go_home()
                continue
            if selection == NAV_BACK:
                go_back()
                continue
            if selection == CLEAR_CACHE_MENU_LABEL:
                run_clear_cache_menu(stdscr)
                continue
            if selection == EXTERNAL_EDITOR_MENU_LABEL:
                current = load_external_editor_command()
                form = show_entry_form(
                    stdscr,
                    "Outside Editor",
                    "Configure default external editor command",
                    [
                        {
                            "key": "outside_editor_command",
                            "label": "Editor command",
                            "default": current,
                            "required": False,
                            "hint": "Used when the app launches an external text editor.",
                        }
                    ],
                    instructions=[
                        "Default is featherpad.",
                        "Set once and it persists across runs; change again anytime.",
                        "Examples: featherpad, gedit, kate, code --wait",
                    ],
                )
                if form is None:
                    continue
                chosen = save_external_editor_command(
                    str(form.get("outside_editor_command") or "")
                )
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                lines = [
                    "Outside editor updated.",
                    f"Current default: {chosen}",
                    "",
                    "Press any key to return to the menu.",
                ]
                for row, line in enumerate(lines):
                    if row >= max_y - 2:
                        break
                    _safe_addstr(stdscr, row, 2, line[: max(1, max_x - 4)])
                stdscr.refresh()
                stdscr.getch()
                continue
            if selection == GATEWAY_GUI_MENU_LABEL:
                already_running = is_gateway_running()
                pid, log_path, err = spawn_gateway_gui()
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                if err:
                    lines = [
                        "Superblock Research Tool (GUI) failed to start.",
                        str(err),
                        "",
                        "Press any key to return to the menu.",
                    ]
                elif already_running:
                    lines = [
                        "Superblock Research Tool (GUI) is already running.",
                        f"PID: {pid}",
                        f"Log: {log_path}",
                        "",
                        "Press any key to return to the menu.",
                    ]
                else:
                    lines = [
                        "Superblock Research Tool (GUI) started in background.",
                        f"PID: {pid}",
                        f"Log: {gateway_log_path()}",
                        "",
                        "The menu stays open — close the GUI window when finished.",
                        "",
                        "Press any key to return to the menu.",
                    ]
                for row, line in enumerate(lines):
                    if row >= max_y - 2:
                        break
                    _safe_addstr(stdscr, row, 2, line[: max(1, max_x - 4)])
                stdscr.refresh()
                stdscr.getch()
                continue
            if selection == RESET_MENU_LABEL:
                if confirm_reset(stdscr, menu_title):
                    tag = resolve_module_tag(menu_title)
                    lines = [f"Reset: {menu_title}", ""]
                    try:
                        if tag is None:
                            lines.append(f"No reset profile for {menu_title!r}.")
                        else:
                            summary = reset_test_set(tag)
                            lines.extend(
                                [
                                    f"Ledgers cleared: {len(summary['ledgers_cleared'])}",
                                    f"Run logs deleted: {len(summary['logs_deleted'])}",
                                    f"Finished archives removed: {len(summary['archives_deleted'])}",
                                    f"Datasets restored: {len(summary['archives_restored'])}",
                                    "",
                                    "Test set is ready to run from the beginning.",
                                ]
                            )
                    except Exception as exc:
                        detail = str(exc).strip()
                        label = type(exc).__name__
                        lines.append(f"Reset failed: {detail or label}")
                    lines.extend(["", "Press any key to return to the menu."])
                    stdscr.clear()
                    max_y, max_x = stdscr.getmaxyx()
                    for row, line in enumerate(lines):
                        if row >= max_y - 2:
                            break
                        _safe_addstr(stdscr, row, 2, line)
                    stdscr.refresh()
                    stdscr.getch()
                continue
            if selection in repos:
                descend_into(selection)
                continue
            if selection in submenus:
                descend_into_submenu(selection)
                continue

            if selection in {DATASET_FETCH_ACTION, DATASET_RUN_ACTION}:
                from sb_shared.dataset_menu import run_dataset_manager

                mode = "fetch" if selection == DATASET_FETCH_ACTION else "run"
                picked = run_dataset_manager(stdscr, mode=mode)
                if picked is None:
                    continue
                module_tag, action, targets, options = picked
                return DATASET_MANAGER_TAG, action, {
                    "dataset_manager_mode": mode,
                    "module_tag": module_tag,
                    "target_ids": targets,
                    **options,
                }

            entry_params = {}
            if selection in module_action_map:
                module_tag = module_action_map[selection]
                params: dict[str, str] = {}

                hook = handle_module_pre_form(module_tag, stdscr, selection, params)
                if hook == "continue":
                    continue
                if hook == "skip_form":
                    return module_tag, selection, params

                fields = module_entry_fields(module_tag, selection)
                fields = strip_auto_filled_fields(fields, params)
                fields = seed_field_defaults_from_params(fields, params)
                entry_params = show_entry_form(
                    stdscr,
                    "Module Entry Form",
                    f"{menu_title} / {selection}",
                    append_llm_entry_fields(fields),
                    instructions=module_entry_instructions(module_tag, selection),
                )
                if entry_params is None:
                    continue
                params = {**params, **entry_params}
                return module_tag, selection, params

            entry_params = show_entry_form(
                stdscr,
                "Data Pull Entry Form",
                f"{menu_title} / {selection}",
                data_pull_entry_fields(selection),
                instructions=data_pull_entry_instructions(selection),
            )
            if entry_params is None:
                continue
            return selection, entry_params.get("query", ""), entry_params

