import os
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


root = _project_root()
if root not in sys.path:
    sys.path.insert(0, root)
