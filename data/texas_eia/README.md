# Texas CI input

Source: [EIA ERCOT workbook](https://www.eia.gov/electricity/gridmonitor/knownissues/xls/ERCO.xlsx),
`Published Hourly Data`, `CO2 Emissions Intensity for Consumed Electricity`.
The `Notes` sheet defines this as lbCO2/kWh and `UTC time` as the hour ending.
Conversion is exactly ×453.59237 to gCO2/kWh, with intervals [hour ending − 1h, hour ending).
This is average consumption-based operational **CO2**, not lifecycle CO2e or marginal emissions.

`ci.csv` covers 2019-01-01 through 2025-01-04 (exclusive). Eleven missing
hours use the last published value, at most two consecutive hours. Each filled
row is labeled and retains its source hour. Longer gaps remain errors.
`source.json` preserves source/output hashes, dates, exact missing hours and rules.

Availability is a declared scenario: 24 hours after each interval ends.
This downloaded archive may contain revisions; the timestamps are not claimed
to be actual historical releases. Queue timestamps still need their own timezone
mapping; no silent calendar shift is applied by the converter.

To reproduce from the same downloaded workbook snapshot:

```bash
"$CARBON_PYTHON" -m carbon prepare-ci --workbook /path/to/ERCO.xlsx \
  --output data/texas_eia-new --max-forward-fill-hours 2
```

EIA updates its workbook, so a later download may have a different hash or values.
The converter needs no Excel installation or additional Python packages.
