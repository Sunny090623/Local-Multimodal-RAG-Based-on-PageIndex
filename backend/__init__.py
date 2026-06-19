"""
Backend package initializer.

Centralizes sys.path configuration so that all backend modules can import
from the project root and the PageIndex package without repeating path hacks.
"""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent
_pageindex = _root / "PageIndex"

# Add project root and PageIndex package to sys.path (idempotent)
for _p in [str(_root), str(_pageindex)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
