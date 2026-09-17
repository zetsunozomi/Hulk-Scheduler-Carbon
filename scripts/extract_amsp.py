#!/usr/bin/env python3
"""Verify the archived AMSP TGS extraction; PDF tools are author-side only.

Run with the writing Python and a local copy of arXiv:2311.00257v2.
This reads vector bars, not screenshots, and never interpolates missing scales.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(pdf_path, record_path):
    import fitz
    record = json.loads(Path(record_path).read_text())
    assert hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest() == record['pdf_sha256'], 'Wrong PDF version'
    doc = fitz.open(pdf_path)
    page = doc[record['page_one_based'] - 1]
    drawings = page.get_drawings()
    for row in record['rows']:
        rect = row['bar_rectangle_pdf_points']
        matching = [d for d in drawings if d['fill'] and abs(d['fill'][0]-.2)<1e-5 and
                    abs(d['fill'][1]-.49777779)<1e-5 and d['fill'][2]>.999 and
                    max(abs(a-b) for a,b in zip(d['rect'], rect))<.001]
        assert len(matching) == 1, (row['model'], row['gpus'], 'bar not found uniquely')
        top,bottom = row['axis_pdf_points']
        value = (bottom-rect[1])*row['axis_tgs_max']/(bottom-top)
        assert round(value/10)*10 == row['tgs_digitized']
    return {'status':'pass','kind':'published_input_extraction_not_experiment',
            'source_pdf_sha256':record['pdf_sha256'],'verified_bars':len(record['rows'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--record', default=str(Path(__file__).resolve().parents[1]/'data/amsp/profiles.json'))
    args = parser.parse_args()
    print(json.dumps(verify(args.pdf,args.record),indent=2))
