"""python -m bbn_isolated.alterbbn {status,ensure,build} [--force] [--dir TREE] [--quiet]

status  print where sb_bbn.x / sb_bbn_full.x resolve, the compiler and the cache paths
ensure  fetch (pinned sha256) + build into datasets/prime_past/alterbbn_v2.2/ if a binary is missing
build   compile both drivers in TREE (default: datasets/prime_past/alterbbn_v2.2/)
Exit code 0 on success, 3 when AlterBBN could not be made available.
"""

from __future__ import annotations

import argparse
import json
import sys

from bbn_isolated import alterbbn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bbn_isolated.alterbbn")
    ap.add_argument("command", choices=("status", "ensure", "build"), nargs="?", default="status")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.command == "status":
            out = alterbbn.status()
        elif args.command == "ensure":
            out = alterbbn.ensure_alterbbn(force=args.force, verbose=not args.quiet)
        else:
            out = {k: str(v) for k, v in alterbbn.build(args.dir or alterbbn.LOCAL_TREE, verbose=not args.quiet).items()}
    except alterbbn.AlterBBNUnavailable as exc:
        print(f"[Superblock Residual Engine] AlterBBN unavailable: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
