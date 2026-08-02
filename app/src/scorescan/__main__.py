from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .self_test import run_system_check


def main() -> None:
    parser = argparse.ArgumentParser(prog="scorescan")
    parser.add_argument("--self-test", action="store_true", help="运行本地完整性与依赖自检后退出")
    parser.add_argument("--json", action="store_true", help="自检时输出紧凑 JSON")
    parser.add_argument("--root", type=Path, help="覆盖便携程序根目录")
    args = parser.parse_args()

    if args.self_test:
        root = (args.root or Path.cwd()).resolve()
        result = run_system_check(Settings.from_root(root))
        print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
        raise SystemExit(0 if result["ok"] else 1)
    # Keep diagnostics usable even when Flask itself is missing or damaged.
    from .server import run_server

    run_server()


if __name__ == "__main__":
    main()
