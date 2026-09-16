from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from carbon.common import timestamp
from carbon.config import Bundle

ROOT = Path(__file__).resolve().parents[1]
T0 = timestamp("2024-01-01T00:00:00Z")


def bundle():
    return Bundle(ROOT / "configs/synthetic.json")


def cluster(nodes=32):
    c = deepcopy(bundle().raw["cluster"])
    c["nodes"] = nodes
    return c
