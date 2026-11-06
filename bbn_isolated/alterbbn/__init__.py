"""AlterBBN v2.2 wrapper: the Prime-past standard-BBN baseline (author ruling 2026-11-04).

AlterBBN (Arbey, Auffinger, Hickerson & Jenssen, arXiv:1806.11095) is run as an external binary:

  sb_bbn.x       Superblock driver shipped with the local AlterBBN tree. Prints
                 Y_p, D/H, 3He/H, 7Li/H, 7Be/H. This is the baseline B.
  sb_bbn_full.x  same argv and physics, built from ``bbn_isolated/alterbbn/sb_bbn_full.c`` by
                 ``build_full_driver.sh``. Prints the whole abundance vector (Y_H, A_i, Z_i) so
                 gate G1 can check baryon-number closure.

argv: ``eta dNnu tau_n none failsafe``. AlterBBN's default Nnu is 3.046 (e± reheating and
non-instantaneous decoupling folded in), so N_eff = 3.044 is ``dNnu = 3.044 - 3.046``.
``failsafe`` picks the integrator precision (0 fast, 1 precise default, 2 / 3 tighter).

Outputs are cached like the other module caches: one CSV per (driver, eta, N_eff, tau_n, failsafe)
under ``datasets/prime_past/alterbbn/`` (gitignored), DataPoint-style columns
``observable/value/unc/source/year/units/notes/reference`` plus the run parameters and the binary's
sha256, rows in a fixed order. A cache row is reused only when the parameters match and, if the
binary is present, its hash matches; with no binary a matching cache is still served (``from_cache``).

Bootstrap (``ensure_alterbbn``): when no binary is found, the pinned upstream tarball
(https://alterbbn.hepforge.org/downloads?f=alterbbn_v2.2.tgz, GPLv3, sha256 46b94253…15c17) is fetched
into ``datasets/prime_past/alterbbn_v2.2.tgz``, verified, unpacked to ``datasets/prime_past/alterbbn_v2.2/``
and both drivers (vendored ``sb_bbn.c`` / ``sb_bbn_full.c``) are compiled there with the first of
``$CC``, cc, gcc, clang. No OpenMP flag is used (Apple Clang has none); the serial build reproduces the
OpenMP build bit-for-bit at the Planck inputs. ``SB_ALTERBBN_NO_AUTO_BUILD=1`` disables fetch/build,
``SB_ALTERBBN_DIR`` points at an existing tree, ``SB_ALTERBBN_TARBALL`` at a local copy of the tarball.
CLI: ``python -m bbn_isolated.alterbbn {status,ensure,build}``.

This module imports nothing from the empirical suite (bbn_isolated rule).
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "datasets" / "prime_past" / "alterbbn"
HERE = Path(__file__).resolve().parent
DRIVER_SOURCES = {"sb_bbn.x": HERE / "sb_bbn.c", "sb_bbn_full.x": HERE / "sb_bbn_full.c"}
DRIVER_SOURCE = DRIVER_SOURCES["sb_bbn_full.x"]
PRIME_PAST_DATA = REPO_ROOT / "datasets" / "prime_past"
LOCAL_TREE = PRIME_PAST_DATA / "alterbbn_v2.2"
TARBALL_PATH = PRIME_PAST_DATA / "alterbbn_v2.2.tgz"
UPSTREAM_URL = "https://alterbbn.hepforge.org/downloads?f=alterbbn_v2.2.tgz"
UPSTREAM_SHA256 = "46b94253cde6afce15ddb9367fa454bac91b25c9a1d3412cc19e512ec3515c17"
UPSTREAM_TOPDIR = "alterbbn_v2.2"
LICENSE = "GPL-3.0-or-later (AlterBBN README)"
_SRC_FILES = ("bbn.c", "bbnrate.c", "bbn_constraints.c", "cosmodel.c", "general.c")
_CFLAGS = ("-O3", "-fomit-frame-pointer", "-ffast-math", "-fno-finite-math-only", "-std=gnu99",
           "-Wno-unknown-pragmas")
BUILD_SCRIPT = Path(__file__).resolve().parent / "build_full_driver.sh"

BINARY_NAME = "sb_bbn.x"
FULL_BINARY_NAME = "sb_bbn_full.x"
ALTERBBN_VERSION = "AlterBBN v2.2"
ALTERBBN_REF = "Arbey, Auffinger, Hickerson & Jenssen, Comput. Phys. Commun. 248 (2020) 106982, arXiv:1806.11095"
ALTERBBN_NNU_DEFAULT = 3.046

# Planck 2018 inputs of the Prime-past baseline (author ruling 2026-11-04).
PLANCK_INPUTS: dict[str, float] = {"eta": 6.114e-10, "n_eff": 3.044, "tau_n": 877.75}
DEFAULT_FAILSAFE = 1

# Five-column sb_bbn.x output, in print order.
OBSERVABLES = ("Y_p", "D_H", "He3_H", "Li7_H", "Be7_H")
_UNITS = {"Y_p": "4*n_He4/n_b", "D_H": "n_D/n_H", "He3_H": "n_He3/n_H (incl. H3)",
          "Li7_H": "n_Li7/n_H (incl. Be7)", "Be7_H": "n_Be7/n_H"}

_SEARCH_DIRS = (
    LOCAL_TREE,
    REPO_ROOT.parent / "bbncodes" / "alterbbn_v2.2",
    Path.home() / "bbncodes" / "alterbbn_v2.2",
)


class AlterBBNUnavailable(FileNotFoundError):
    """No AlterBBN binary and no matching cached run."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def auto_build_enabled() -> bool:
    return not _truthy(os.environ.get("SB_ALTERBBN_NO_AUTO_BUILD"))


def find_compiler() -> str | None:
    """``$CC``, then cc, gcc, clang."""
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand:
            exe = shutil.which(cand.split()[0])
            if exe:
                return exe
    return None


def _say(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[Superblock Residual Engine] AlterBBN: {msg}")


def fetch_source(*, force: bool = False, verbose: bool = True, timeout: float = 120.0) -> Path:
    """Pinned upstream tarball → ``datasets/prime_past/alterbbn_v2.2.tgz`` (sha256 checked)."""
    candidates = [Path(p).expanduser() for p in (os.environ.get("SB_ALTERBBN_TARBALL"),) if p]
    candidates.append(TARBALL_PATH)
    for cand in candidates:
        if cand.is_file() and not force and _sha256(cand) == UPSTREAM_SHA256:
            _say(verbose, f"source tarball verified at {cand}")
            return cand
    PRIME_PAST_DATA.mkdir(parents=True, exist_ok=True)
    part = TARBALL_PATH.with_suffix(".tgz.part")
    _say(verbose, f"downloading {UPSTREAM_URL}")
    req = urllib.request.Request(UPSTREAM_URL, headers={"User-Agent": "research-tool/alterbbn-bootstrap"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise AlterBBNUnavailable(f"AlterBBN download failed ({exc}); offline? set SB_ALTERBBN_TARBALL") from exc
    digest = _sha256(part)
    if digest != UPSTREAM_SHA256:
        part.unlink(missing_ok=True)
        raise AlterBBNUnavailable(f"AlterBBN tarball sha256 mismatch ({digest}); not used")
    part.replace(TARBALL_PATH)
    _say(verbose, f"tarball cached: {TARBALL_PATH} (sha256 ok)")
    return TARBALL_PATH


def unpack_source(tarball: Path, *, verbose: bool = True) -> Path:
    """Unpack into ``datasets/prime_past/`` (members must stay under ``alterbbn_v2.2/``)."""
    PRIME_PAST_DATA.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if not parts or parts[0] != UPSTREAM_TOPDIR or ".." in parts or Path(m.name).is_absolute():
                raise AlterBBNUnavailable(f"unexpected tarball member {m.name!r}")
            if not (m.isfile() or m.isdir()):
                continue
            members.append(m)
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(PRIME_PAST_DATA, members=members, **kw)
    _say(verbose, f"source unpacked: {LOCAL_TREE}")
    return LOCAL_TREE


def build(tree: str | Path = LOCAL_TREE, *, compiler: str | None = None, verbose: bool = True) -> dict[str, Path]:
    """Compile sb_bbn.x and sb_bbn_full.x in ``tree`` (needs ``tree/src``). No OpenMP."""
    tree = Path(tree)
    src = tree / "src"
    if not (src / "include.h").is_file():
        raise AlterBBNUnavailable(f"AlterBBN src/ not found under {tree}")
    cc = compiler or find_compiler()
    if cc is None:
        raise AlterBBNUnavailable("no C compiler found ($CC, cc, gcc, clang)")
    out: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        (tmpd / "src").symlink_to(src.resolve(), target_is_directory=True)
        for exe_name, driver in DRIVER_SOURCES.items():
            shutil.copy2(driver, tmpd / driver.name)
            target = tmpd / exe_name
            cmd = [cc, *_CFLAGS, f"-I{src}", "-o", str(target), str(tmpd / driver.name),
                   *[str(src / f) for f in _SRC_FILES], "-lm"]
            _say(verbose, f"compiling {exe_name} with {Path(cc).name}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise AlterBBNUnavailable(f"compiling {exe_name} failed: {proc.stderr.strip()[-600:]}")
            dest = tree / exe_name
            shutil.copy2(target, dest)
            os.chmod(dest, 0o755)
            out[exe_name] = dest
    _say(verbose, f"built {', '.join(str(p) for p in out.values())}")
    return out


def status() -> dict[str, Any]:
    return {
        "sb_bbn.x": str(find_binary(BINARY_NAME) or ""),
        "sb_bbn_full.x": str(find_binary(FULL_BINARY_NAME) or ""),
        "local_tree": str(LOCAL_TREE),
        "tarball": str(TARBALL_PATH) if TARBALL_PATH.is_file() else "",
        "compiler": find_compiler() or "",
        "auto_build": auto_build_enabled(),
    }


def ensure_alterbbn(*, force: bool = False, auto_build: bool | None = None, verbose: bool = True) -> dict[str, Any]:
    """Make sb_bbn.x and sb_bbn_full.x available; fetch + build into datasets/prime_past/ if needed.

    Returns ``status()`` plus ``fetched`` / ``built`` flags. Raises AlterBBNUnavailable when the binaries
    are missing and auto-build is disabled, offline, or no compiler exists.
    """
    have = find_binary(BINARY_NAME) is not None and find_binary(FULL_BINARY_NAME) is not None
    if have and not force:
        return {**status(), "fetched": False, "built": False}
    allowed = auto_build_enabled() if auto_build is None else bool(auto_build)
    if not allowed:
        raise AlterBBNUnavailable("AlterBBN binaries not found and auto-build is disabled (SB_ALTERBBN_NO_AUTO_BUILD)")
    fetched = False
    if force or not (LOCAL_TREE / "src" / "include.h").is_file():
        unpack_source(fetch_source(force=False, verbose=verbose), verbose=verbose)
        fetched = True
    build(LOCAL_TREE, verbose=verbose)
    return {**status(), "fetched": fetched, "built": True}


_BOOTSTRAP_TRIED = False


def ensure_available(*, verbose: bool = False) -> bool:
    """True when sb_bbn.x and sb_bbn_full.x can be used; tries one lazy bootstrap per process."""
    global _BOOTSTRAP_TRIED
    if find_binary(BINARY_NAME) is not None and find_binary(FULL_BINARY_NAME) is not None:
        return True
    if _BOOTSTRAP_TRIED or not auto_build_enabled():
        return False
    _BOOTSTRAP_TRIED = True
    try:
        ensure_alterbbn(verbose=verbose)
    except (AlterBBNUnavailable, OSError, subprocess.SubprocessError) as exc:
        _say(verbose, f"bootstrap skipped: {exc}")
        return False
    return find_binary(BINARY_NAME) is not None and find_binary(FULL_BINARY_NAME) is not None


def find_binary(name: str = BINARY_NAME) -> Path | None:
    """``$SB_ALTERBBN_DIR``, then the known AlterBBN trees, then ``$PATH``."""
    dirs: list[Path] = []
    env_dir = os.environ.get("SB_ALTERBBN_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    dirs.extend(_SEARCH_DIRS)
    for d in dirs:
        p = d / name
        if p.is_file() and os.access(p, os.X_OK):
            return p
    hit = shutil.which(name)
    return Path(hit) if hit else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_file(cache_dir: Path, driver: str, eta: float, n_eff: float, tau_n: float, failsafe: int) -> Path:
    stem = driver.replace(".x", "")
    return cache_dir / f"alterbbn__{stem}__eta{eta:.4e}_neff{n_eff:.4f}_tau{tau_n:.2f}_fs{int(failsafe)}.csv"


def _argv(binary: Path, eta: float, n_eff: float, tau_n: float, failsafe: int) -> list[str]:
    dnnu = n_eff - ALTERBBN_NNU_DEFAULT
    return [str(binary), f"{eta:.6e}", f"{dnnu:.6f}", f"{tau_n:.6f}", "none", str(int(failsafe))]


def _parse_short(text: str) -> dict[str, Any]:
    vals = [float(v) for v in text.split()]
    if len(vals) != len(OBSERVABLES):
        raise RuntimeError(f"unexpected {BINARY_NAME} output: {text!r}")
    return {"err": 0, "values": dict(zip(OBSERVABLES, vals))}


def _parse_full(text: str) -> dict[str, Any]:
    tok = text.split()
    err, n = int(tok[0]), int(tok[1])
    need = 2 + (n + 1) + 2 * n
    if len(tok) != need:
        raise RuntimeError(f"unexpected {FULL_BINARY_NAME} output ({len(tok)} tokens, need {need})")
    ratio = [float(v) for v in tok[2:3 + n]]
    A = [0.0] + [float(v) for v in tok[3 + n:3 + 2 * n]]
    Z = [0.0] + [float(v) for v in tok[3 + 2 * n:]]
    values = {"Y_p": ratio[6], "D_H": ratio[3], "He3_H": ratio[5], "Li7_H": ratio[8], "Be7_H": ratio[9]}
    return {"err": err, "values": values, "ratioH": ratio, "A": A, "Z": Z}


def baryon_closure(full: dict[str, Any]) -> float:
    """Σ A_i Y_i − 1 from the full vector (Y_i = ratioH_i · Y_H; ratioH_6 is already 4·Y_He4).

    AlterBBN folds H3 into He3 (i=5) and Be7 into Li7 (i=8) after BBN; both pairs share A, so i=4 and
    i=9 are skipped to avoid double counting.
    """
    r, A = full["ratioH"], full["A"]
    y_h = r[2]
    total = y_h + r[6]
    for i in range(1, len(A)):
        if i in (2, 4, 6, 9):
            continue
        total += A[i] * r[i] * y_h
    return float(total - 1.0)


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "source": ALTERBBN_VERSION,
        "year": 2026,
        "reference": ALTERBBN_REF,
        "eta": result["eta"],
        "n_eff": result["n_eff"],
        "tau_n": result["tau_n"],
        "failsafe": result["failsafe"],
        "driver": result["driver"],
        "binary_sha256": result["binary_sha256"],
        "err": result["err"],
        "A": "",
        "Z": "",
    }
    rows = [
        {**base, "observable": k, "value": result["values"][k], "unc": "", "units": _UNITS[k], "notes": ""}
        for k in OBSERVABLES
    ]
    if "ratioH" in result:
        for i, v in enumerate(result["ratioH"]):
            rows.append({
                **base, "observable": f"ratioH_{i}", "value": v, "unc": "", "units": "",
                "notes": "0=eta, 2=Y_H, 6=4*Y_He4, else n_i/n_H",
                "A": result["A"][i] if i else "", "Z": result["Z"][i] if i else "",
            })
    return rows


_FIELDS = ("observable", "value", "unc", "source", "year", "units", "notes", "reference",
           "eta", "n_eff", "tau_n", "failsafe", "driver", "binary_sha256", "err", "A", "Z")


def _write_cache(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for row in _rows(result):
            w.writerow(row)


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None
    head = rows[0]
    out: dict[str, Any] = {
        "eta": float(head["eta"]), "n_eff": float(head["n_eff"]), "tau_n": float(head["tau_n"]),
        "failsafe": int(head["failsafe"]), "driver": head["driver"], "binary_sha256": head["binary_sha256"],
        "err": int(head["err"]), "values": {},
    }
    ratio, A, Z = [], [0.0], [0.0]
    for row in rows:
        name = row["observable"]
        if name in OBSERVABLES:
            out["values"][name] = float(row["value"])
        elif name.startswith("ratioH_"):
            ratio.append(float(row["value"]))
            if row["A"] != "":
                A.append(float(row["A"]))
                Z.append(float(row["Z"]))
    if set(out["values"]) != set(OBSERVABLES):
        return None
    if ratio:
        out.update(ratioH=ratio, A=A, Z=Z)
    return out


def run(
    eta: float = PLANCK_INPUTS["eta"],
    n_eff: float = PLANCK_INPUTS["n_eff"],
    tau_n: float = PLANCK_INPUTS["tau_n"],
    failsafe: int = DEFAULT_FAILSAFE,
    *,
    full: bool = False,
    binary: str | Path | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """One AlterBBN run (cached). ``full=True`` uses the full-vector driver."""
    name = FULL_BINARY_NAME if full else BINARY_NAME
    exe = Path(binary) if binary else find_binary(name)
    if exe is None and binary is None and auto_build_enabled():
        ensure_available(verbose=True)
        exe = find_binary(name)
    if exe is not None and not exe.is_file():
        exe = None
    cdir = Path(cache_dir) if cache_dir else CACHE_DIR
    cpath = _cache_file(cdir, name, eta, n_eff, tau_n, failsafe)
    digest = _sha256(exe) if exe is not None else None
    if use_cache and cpath.is_file():
        cached = _read_cache(cpath)
        if cached is not None and (digest is None or cached["binary_sha256"] == digest):
            cached.update(from_cache=True, cache_path=str(cpath), binary=str(exe) if exe else None)
            return cached
    if exe is None:
        raise AlterBBNUnavailable(
            f"AlterBBN binary {name} not found (set SB_ALTERBBN_DIR, or run "
            f"'python -m bbn_isolated.alterbbn ensure') and no cached run at {cpath}"
        )
    env = {k: v for k, v in os.environ.items() if k != "SB_KAPPA"}  # no lever in the baseline
    proc = subprocess.run(
        _argv(exe, eta, n_eff, tau_n, failsafe), capture_output=True, text=True, timeout=timeout,
        check=False, env=env, cwd=tempfile.gettempdir(),
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"{name} exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    parsed = _parse_full(proc.stdout) if full else _parse_short(proc.stdout)
    result: dict[str, Any] = {
        "eta": float(eta), "n_eff": float(n_eff), "tau_n": float(tau_n), "failsafe": int(failsafe),
        "driver": name, "binary_sha256": digest, **parsed,
    }
    if not all(math.isfinite(v) for v in result["values"].values()):
        result["err"] = result.get("err") or -1
    if use_cache:
        _write_cache(cpath, result)
    result.update(from_cache=False, cache_path=str(cpath) if use_cache else None, binary=str(exe))
    return result


def available() -> bool:
    return find_binary(BINARY_NAME) is not None


__all__ = [
    "ALTERBBN_REF",
    "ALTERBBN_VERSION",
    "AlterBBNUnavailable",
    "CACHE_DIR",
    "OBSERVABLES",
    "PLANCK_INPUTS",
    "LOCAL_TREE",
    "UPSTREAM_SHA256",
    "UPSTREAM_URL",
    "auto_build_enabled",
    "available",
    "baryon_closure",
    "build",
    "ensure_alterbbn",
    "ensure_available",
    "fetch_source",
    "find_binary",
    "find_compiler",
    "run",
    "status",
]
