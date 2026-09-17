from datetime import timedelta
from decimal import Decimal
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.sax.saxutils import escape
from zipfile import ZipFile

from carbon.carbon import CarbonSeries, CIRecord, Exposure
from carbon.common import ContractError, timestamp
from carbon.inputs import INTENSITY, NS, eia_rows, prepare_eia

T0 = timestamp("2024-01-01T00:00:00Z")


class InputTests(unittest.TestCase):
    def fixture(self, directory, unit="lbs/kWh", timestamp_note="The end of the hour in UTC"):
        path = Path(directory) / "synthetic.xlsx"
        def textcell(address, text):
            return f'<c r="{address}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
        def sheet(rows):
            return '<worksheet xmlns="' + NS[1:-1] + '"><sheetData>' + rows + '</sheetData></worksheet>'
        notes = sheet('<row r="1">' + textcell('A1', INTENSITY) + textcell('B1', unit) + '</row>' +
                      '<row r="2">' + textcell('A2', 'UTC time') + textcell('B2', timestamp_note) + '</row>')
        hour = sheet('<row r="1">' + textcell('A1', 'BA') + textcell('B1', 'UTC time') +
                     textcell('C1', INTENSITY) + '</row><row r="2">' + textcell('A2', 'ERCO') +
                     '<c r="B2"><v>45292.04166666667</v></c><c r="C2"><v>1</v></c></row>')
        with ZipFile(path, 'w') as z:
            z.writestr('xl/workbook.xml', '<workbook xmlns="' + NS[1:-1] +
                       '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                       '<sheets><sheet name="Notes" r:id="r1"/><sheet name="Published Hourly Data" r:id="r2"/></sheets></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="r1" Target="worksheets/s1.xml"/>'
                       '<Relationship Id="r2" Target="worksheets/s2.xml"/></Relationships>')
            z.writestr('xl/worksheets/s1.xml', notes)
            z.writestr('xl/worksheets/s2.xml', hour)
        return path

    def test_pounds_conversion_hour_ending_and_release_lag(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.fixture(d)
            records = list(eia_rows(path))
            self.assertEqual(records, [(T0 + timedelta(hours=1), Decimal('453.59237'))])
            out = Path(d) / 'converted'
            report = prepare_eia(path, out, '2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z')
            self.assertTrue(report['coverage_complete'])
            config = dict(unit='gCO2/kWh', source='synthetic test', region='test', queue_to_ci_offset_seconds=0)
            ci = CarbonSeries.load(out / 'ci.csv', config)
            self.assertEqual(ci.integral(T0, T0 + timedelta(hours=1)), Decimal('453.59237'))
            self.assertEqual(ci.observations_available_at(T0 + timedelta(hours=24)), ())
            self.assertEqual(len(ci.observations_available_at(T0 + timedelta(hours=25))), 1)
            with self.assertRaisesRegex(ContractError, 'already exists'):
                prepare_eia(path, out, '2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z')

    def test_changed_source_unit_or_timestamp_semantics_fail(self):
        for options in ({'unit': 'kg/kWh'}, {'timestamp_note': 'start of hour'}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ContractError):
                    list(eia_rows(self.fixture(d, **options)))

    def test_missing_hours_are_not_silently_interpolated(self):
        rows = [(T0 + timedelta(hours=i+1), None if i in (1, 2, 3) else Decimal(i+10)) for i in range(5)]
        with tempfile.TemporaryDirectory() as d, patch('carbon.inputs.eia_rows', return_value=iter(rows)):
            path = self.fixture(d)
            report = prepare_eia(path, Path(d)/'out', '2024-01-01T00:00:00Z', '2024-01-01T05:00:00Z')
            self.assertEqual(report['retained_hours'], 2)
            self.assertFalse(report['coverage_complete'])
            self.assertEqual(report['forward_filled_hours'], [])

    def test_forward_fill_uses_only_past_values_and_is_bounded(self):
        rows = [(T0 + timedelta(hours=i+1), None if i in (1, 2, 3) else Decimal(i+10)) for i in range(5)]
        with tempfile.TemporaryDirectory() as d, patch('carbon.inputs.eia_rows', return_value=iter(rows)):
            path = self.fixture(d); out = Path(d)/'out'
            report = prepare_eia(path, out, '2024-01-01T00:00:00Z', '2024-01-01T05:00:00Z', max_forward_fill_hours=2)
            self.assertEqual(len(report['forward_filled_hours']), 2)
            self.assertFalse(report['coverage_complete'])
            with (out/'ci.csv').open() as f:
                values = list(csv.DictReader(f))
            self.assertEqual([v['gco2_per_kwh'] for v in values], ['10', '10', '10', '14'])
            self.assertEqual(values[2]['value_source_end_utc'], '2024-01-01T01:00:00Z')
            self.assertEqual(values[2]['value_status'], 'forward_filled')

    def test_co2_is_not_mislabeled_as_co2e_in_phase_logs(self):
        ci = CarbonSeries([CIRecord(T0, T0+timedelta(hours=1), Decimal(100), T0)], 'synthetic', 'test', unit='gCO2/kWh')
        record = Exposure([4]).add(4, [('training', T0, T0+timedelta(hours=1))], ci)[0]
        self.assertEqual(record['exposure_node_gco2_per_kwh_hours'], 400)
        self.assertNotIn('exposure_node_gco2e_per_kwh_hours', record)
        self.assertEqual(record['ci_unit'], 'gCO2/kWh')

    def test_input_pollutant_column_must_match_declared_unit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'ci.csv'
            path.write_text('start_utc,end_utc,gco2e_per_kwh,available_at_utc\n2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,400,2024-01-01T01:00:00Z\n')
            with self.assertRaisesRegex(ContractError, 'gco2_per_kwh'):
                CarbonSeries.load(path, dict(unit='gCO2/kWh', source='test', region='test', queue_to_ci_offset_seconds=0))
