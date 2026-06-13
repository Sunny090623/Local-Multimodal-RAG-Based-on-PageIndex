"""
Shared singleton PageIndexClient instance.

All modules (parser.py, rag.py, app.py) should use get_shared_client()
instead of creating their own PageIndexClient instances. This ensures:
1. Data consistency — single in-memory document registry
2. No concurrent write conflicts on _meta.json
3. No redundant disk reads on every API request
"""

import sys
from pathlib import Path

# Ensure PageIndex package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "PageIndex"))

from pageindex import PageIndexClient

# Base directories
STORAGE_DIR = Path(__file__).parent / "storage"
WORKSPACE_DIR = STORAGE_DIR / "workspace"
IMAGES_DIR = STORAGE_DIR / "images"

for d in [STORAGE_DIR, WORKSPACE_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Singleton instance
_client = None


def get_shared_client() -> PageIndexClient:
    """Return the global PageIndexClient singleton, creating it on first call."""
    global _client
    if _client is None:
        _client = PageIndexClient(workspace=str(WORKSPACE_DIR))
    return _client
