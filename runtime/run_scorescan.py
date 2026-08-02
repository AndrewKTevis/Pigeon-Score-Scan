from __future__ import annotations

"""Start the application from the bundled, relocatable offline runtime."""

import runpy
import site
import sys
from pathlib import Path


runtime_root = Path(__file__).resolve().parent
product_root = runtime_root.parent
site_packages = runtime_root / "site-packages"
application_source = product_root / "app" / "src"

if not site_packages.is_dir() or not application_source.is_dir():
    raise SystemExit("The bundled Pigeon Score Scan runtime is incomplete.")

site.addsitedir(str(site_packages))
sys.path.insert(0, str(application_source))
runpy.run_module("scorescan", run_name="__main__", alter_sys=True)
