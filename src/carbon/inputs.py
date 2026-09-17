"""Import published EIA ERCOT CI; no scheduler replay or workload assumptions."""

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .common import digest, iso, json_text, number, require, timestamp

EIA_URL = "https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ERCO.xlsx"
INTENSITY = "CO2 Emissions Intensity for Consumed Electricity"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def workbook_rows(path, sheet_name):
    """Stream cached XLSX values with stdlib only; never execute workbook formulas."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as handle:
                for _, element in ET.iterparse(handle, events=("end",)):
                    if element.tag == NS + "si":
                        strings.append("".join(t.text or "" for t in element.iter(NS + "t")))
                        element.clear()
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        props = book.find(NS + "workbookPr")
        require(props is None or props.get("date1904", "0") in {"0", "false"},
                "Unsupported XLSX 1904 date system")
        sheets = {s.get("name"): s.get(REL) for s in book.find(NS + "sheets")}
        require(sheet_name in sheets, f"Missing EIA worksheet: {sheet_name}")
        links = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in links}
        target = targets[sheets[sheet_name]]
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        with archive.open(target) as handle:
            row_parent = None
            for event, element in ET.iterparse(handle, events=("start", "end")):
                if event == "start" and element.tag == NS + "sheetData":
                    row_parent = element
                if event != "end" or element.tag != NS + "row":
                    continue
                values = {}
                for cell in element.findall(NS + "c"):
                    col = "".join(c for c in cell.get("r", "") if c.isalpha())
                    value = cell.find(NS + "v")
                    raw = value.text if value is not None else None
                    if cell.get("t") == "s" and raw is not None:
                        raw = strings[int(raw)]
                    elif cell.get("t") == "inlineStr":
                        raw = "".join(t.text or "" for t in cell.iter(NS + "t"))
                    elif cell.get("t") == "e":
                        raise ValueError(f"Excel error at {sheet_name}!{cell.get('r')}: {raw}")
                    values[col] = raw
                yield values
                element.clear()
                if row_parent is not None:
                    row_parent.remove(element)


def eia_rows(path):
    """Return UTC hour endings and gCO2/kWh, verifying source definitions first."""
    notes = {}
    for row in workbook_rows(path, "Notes"):
        if row.get("A") and row.get("B"):
            notes[row["A"].strip()] = row["B"].strip()
    require("lbs/kWh" in notes.get(INTENSITY, ""), "EIA intensity unit changed; audit the source")
    require("end of the hour" in notes.get("UTC time", ""), "EIA timestamp meaning changed")
    rows = iter(workbook_rows(path, "Published Hourly Data"))
    columns = {value: col for col, value in next(rows).items()}
    require({"BA", "UTC time", INTENSITY} <= columns.keys(), "EIA hourly columns changed")
    epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
    for row in rows:
        if row.get(columns["BA"]) != "ERCO":
            continue
        raw_time = row.get(columns["UTC time"])
        require(raw_time is not None, "EIA row has no UTC timestamp")
        sec = Decimal(raw_time) * 86400
        # Excel stores day fractions as floats; the published series is hourly.
        rounded = int((sec / 3600).to_integral_value()) * 3600
        require(abs(sec - rounded) < Decimal("0.01"), "EIA timestamp is not an hour boundary")
        end = epoch + timedelta(seconds=rounded)
        raw = row.get(columns[INTENSITY])
        value = None if raw is None else number(raw, "EIA intensity") * Decimal("453.59237")
        yield end, value


def prepare_eia(path, output, start, end, release_lag_hours=24, max_forward_fill_hours=0):
    """Preserve gaps unless a bounded past-only fill was explicitly requested."""
    path, output = Path(path), Path(output)
    start, end = timestamp(start), timestamp(end)
    require(start < end, "CI start must precede end")
    require(all(t.minute == t.second == t.microsecond == 0 for t in (start, end)),
            "CI range must use whole UTC hours")
    lag = number(release_lag_hours, "release_lag_hours")
    require(isinstance(max_forward_fill_hours, int) and 0 <= max_forward_fill_hours <= 2,
            "Only an explicit 0, 1, or 2 hour causal forward fill is supported")
    require(not output.exists(), f"Output already exists: {output}")
    source_hash = digest(path)
    retained, missing, seen = [], [], set()
    for stop, value in eia_rows(path):
        begin = stop - timedelta(hours=1)
        if not start <= begin < end:
            continue
        require(begin not in seen, f"Duplicate EIA UTC hour {iso(begin)}")
        seen.add(begin)
        if value is None:
            missing.append(iso(begin))
        else:
            retained.append((begin, stop, value, stop + timedelta(seconds=float(lag * 3600))))
    require(retained, "No nonmissing EIA CI in selected interval")
    expected = int((end - start).total_seconds() / 3600)
    absent = [iso(start + timedelta(hours=i)) for i in range(expected)
              if start + timedelta(hours=i) not in seen]
    by_start = {r[0]: r for r in retained}
    retained, filled = [], []
    previous = None
    for i in range(expected):
        begin = start + timedelta(hours=i)
        record = by_start.get(begin)
        if record is not None:
            previous = record
            retained.append((*record, "published", record[1]))
        elif previous is not None and begin - previous[0] <= timedelta(hours=max_forward_fill_hours):
            stop = begin + timedelta(hours=1)
            retained.append((begin, stop, previous[2], stop + timedelta(seconds=float(lag * 3600)),
                             "forward_filled", previous[1]))
            filled.append(iso(begin))
    segments = []
    for begin, stop, *_ in retained:
        if segments and segments[-1][1] == iso(begin):
            segments[-1][1] = iso(stop)
        else:
            segments.append([iso(begin), iso(stop)])
    output.mkdir(parents=True)
    csv_path = output / "ci.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_utc", "end_utc", "gco2_per_kwh", "available_at_utc", "value_status", "value_source_end_utc"])
        writer.writerows((iso(a), iso(b), str(v), iso(r), status, iso(source_end))
                         for a, b, v, r, status, source_end in retained)
    report = {
        "converted_at_utc": iso(datetime.now(timezone.utc)),
        "source_url": EIA_URL, "source_sha256": source_hash,
        "source_worksheet": "Published Hourly Data", "source_column": INTENSITY,
        "source_unit": "lbCO2/kWh", "unit": "gCO2/kWh", "pollutant": "CO2",
        "conversion": "source_value * 453.59237", "region": "ERCOT (ERCO), Texas",
        "type": "average_operational", "boundary": "consumption-based",
        "time_rule": "UTC time is the hour ending; interval is [UTC time - 1h, UTC time)",
        "availability_rule": f"Scenario: released {lag} hours after interval end; revised archive, not vintage forecasts",
        "actual_publication_timestamps_known": False,
        "release_lag_hours": float(lag), "requested_interval": [iso(start), iso(end)],
        "retained_hours": len(retained), "expected_hours": expected,
        "missing_value_hours": missing, "absent_hours": absent,
        "missing_value_rule": f"Forward-fill at most {max_forward_fill_hours} consecutive hours from past published CI; retain longer gaps",
        "max_forward_fill_hours": max_forward_fill_hours, "forward_filled_hours": filled,
        "coverage_complete": len(retained) == expected, "continuous_segments": segments,
        "minimum_gco2_per_kwh": float(min(r[2] for r in retained)),
        "maximum_gco2_per_kwh": float(max(r[2] for r in retained)),
        "ci_sha256": digest(csv_path),
        "not_included": "non-CO2 greenhouse gases, lifecycle emissions, node power measurements",
    }
    (output / "source.json").write_text(json_text(report), encoding="utf-8")
    return report
