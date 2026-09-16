"""Small, dependency-free validation and time helpers. All internal time is UTC."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc


class ContractError(ValueError):
    """Invalid input or violation of the replay contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def number(value, name, minimum=0, strict=False):
    require(not isinstance(value, bool), f"{name}: expected a number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ContractError(f"{name}: expected a finite number, got {value!r}") from None
    require(result.is_finite(), f"{name}: must be finite")
    require(result > minimum if strict else result >= minimum,
            f"{name}: must be {'>' if strict else '>='} {minimum}")
    return result


def integer(value, name, minimum=1):
    result = number(value, name, minimum)
    require(result == result.to_integral_value(), f"{name}: expected an integer")
    return int(result)


def timestamp(value, local_zone=None, fold=None):
    require(isinstance(value, str), f"Timestamp must be a string: {value!r}")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ContractError(f"Invalid timestamp: {value!r}") from None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC)
    require(local_zone is not None, f"Timestamp needs a UTC offset: {value}")
    zone = ZoneInfo(local_zone)
    candidates = []
    for f in (0, 1):
        candidate = dt.replace(tzinfo=zone, fold=f)
        if candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == dt:
            candidates.append(candidate)
    require(candidates, f"Nonexistent local time in {local_zone}: {value}")
    ambiguous = len({c.utcoffset() for c in candidates}) > 1
    require(not ambiguous or fold in (0, 1),
            f"Ambiguous local time in {local_zone}: {value}; declare dst_fold or use offsets")
    chosen = dt.replace(tzinfo=zone, fold=fold or 0)
    return chosen.astimezone(UTC)


def iso(dt):
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z") if dt is not None else None


def seconds(delta):
    return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000


def duration(value, name="seconds"):
    microseconds = number(value, name) * 1000000
    require(microseconds == microseconds.to_integral_value(), f"{name}: sub-microsecond precision unsupported")
    return timedelta(microseconds=int(microseconds))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def load_json(path):
    def reject_constant(value):
        raise ContractError(f"Non-finite JSON constant: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_keys)
