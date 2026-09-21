#!/Users/shuyuanfan/miniconda3/envs/writing/bin/python
"""Check the draft and independently verify its algebra with synthetic inputs.

This is manuscript QA, not paper evaluation. It writes no experimental results.
Run latexmk first. Use the user's writing environment for PDF inspection.
"""
from pathlib import Path
import itertools
import json
import math
import random
import re
import fitz

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ["main.tex", "1intro.tex", "3related.tex", "4Jobtrace.tex",
            "6design.tex", "5algorithm.tex", "7exp.tex", "8conclusion.tex"]
text = "\n".join((ROOT / p).read_text() for p in SECTIONS)
labels = re.findall(r"\\label\{([^}]+)\}", text)
refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
assert len(labels) == len(set(labels)), "Duplicate LaTeX label"
assert set(refs) <= set(labels), f"Missing labels: {set(refs)-set(labels)}"
cites = set()
for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
    cites.update(x.strip() for x in group.split(","))
bib = "\n".join((ROOT / p).read_text() for p in ["reference.bib", "scaledown_refs.bib"])
keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
assert cites <= keys, f"Missing citations: {cites-keys}"
assert r"\includegraphics" not in text, "Old result figures must not enter this draft"
assert "XXX" not in text, "Use named evidence slots, not stale XXX"
for evidence in ("E1", "E2", "E3", "E4"):
    assert evidence in text
for stale in ("gamma=0.99", "measured or calibrated", "scale-down intensity"):
    assert stale not in text, f"Stale design: {stale}"

# Check formulas against explicit work allocation / interval exposure / box vertices.
rng = random.Random(20260915)
scales = (4, 16, 64, 128)
cases = 1000
for _ in range(cases):
    q = [rng.uniform(0.5, 8) for _ in scales]
    total = rng.randint(30, 20000)
    parts = [rng.random() for _ in scales]
    parts = [x / sum(parts) for x in parts]
    p = [rng.uniform(0.1, 3) for _ in scales]
    ci = rng.uniform(0.01, 0.9)
    fixed_t = [total / x for x in q]
    fixed_c = [n * pw * t * ci for n, pw, t in zip(scales, p, fixed_t)]
    actual_t = sum(total * x / rate for x, rate in zip(parts, q))
    actual_c = sum(n * pw * total * x / rate * ci
                   for n, pw, x, rate in zip(scales, p, parts, q))
    assert math.isclose(actual_t, sum(x*t for x,t in zip(parts,fixed_t)), rel_tol=1e-12)
    assert math.isclose(actual_c, sum(x*c for x,c in zip(parts,fixed_c)), rel_tol=1e-12)

    eta = [(4/q[0])/(n/rate) for n,rate in zip(scales,q)]
    # Independent irregular intervals at each scale, integrated as piecewise constant CI.
    log_a = [[(rng.uniform(.05, 10), rng.uniform(.01, 1))
              for _ in range(rng.randint(1, 8))] for n in scales]
    log_b = [[(rng.uniform(.05, 10), rng.uniform(.01, 1))
              for _ in range(rng.randint(1, 8))] for n in scales]
    la = [n*sum(dt*c for dt,c in logs) for n,logs in zip(scales,log_a)]
    lb = [n*sum(dt*c for dt,c in logs) for n,logs in zip(scales,log_b)]
    da = sum(la)-sum(lb)
    db = sum(e*(a-b) for e,a,b in zip(eta,la,lb))
    lo, hi = sorted([rng.random(), rng.random()])
    coef = rng.uniform(.01, 5)
    ratios, diffs = [], []
    for j in range(41):
        rho = lo+(hi-lo)*j/40
        sn = [rho+(1-rho)*e for e in eta]
        direct_a = coef*sum(n*s*dt*c for n,s,logs in zip(scales,sn,log_a)
                            for dt,c in logs)
        direct_b = coef*sum(n*s*dt*c for n,s,logs in zip(scales,sn,log_b)
                            for dt,c in logs)
        assert math.isclose(direct_a-direct_b, coef*(db+rho*(da-db)),
                            rel_tol=1e-9, abs_tol=1e-10)
        ratios.append(direct_a/direct_b)
        diffs.append(direct_a-direct_b)
    assert max(diffs) <= max(diffs[0], diffs[-1])+1e-8
    assert max(ratios) <= max(ratios[0], ratios[-1])+1e-8

    rho = rng.random()
    sn = [rho+(1-rho)*e for e in eta]
    dl = [a-b for a,b in zip(la,lb)]
    radius = rng.random()*.9
    vertices = []
    for signs in itertools.product((-1, 1), repeat=3):
        powers = [1.0]+[s*(1+radius*sign) for s,sign in zip(sn[1:],signs)]
        vertices.append(sum(power*d for power,d in zip(powers,dl)))
    bound = sum(s*d for s,d in zip(sn,dl))+radius*sum(s*abs(d) for s,d in zip(sn[1:],dl[1:]))
    assert math.isclose(max(vertices), bound, rel_tol=1e-10, abs_tol=1e-10)

    wpi = sum(n*total*x/rate for n,x,rate in zip(scales,parts,q))
    w4 = 4*total/q[0]
    direct_e = sum(n*(rho+(1-rho)*e)*total*x/rate
                   for n,e,x,rate in zip(scales,eta,parts,q))
    assert math.isclose(direct_e, rho*wpi+(1-rho)*w4, rel_tol=1e-12)

# Check the compiled artifact, not merely the source.
log = (ROOT/"main.log").read_text(errors="replace")
fatal_patterns = ("Overfull", "undefined", "LaTeX Error", "Emergency stop", "multiply defined")
for pattern in fatal_patterns:
    assert pattern not in log, f"Build issue: {pattern}"
pdf = fitz.open(ROOT/"main.pdf")
reference_page = None
for i, page in enumerate(pdf):
    content = page.get_text()
    if "References" in content and reference_page is None:
        reference_page = i+1
    assert "\ufffd" not in content, f"Replacement character on page {i+1}"
    for x0,y0,x1,y1,*_ in page.get_text("blocks"):
        assert x0 >= 40 and x1 <= page.rect.width-40, f"Text outside horizontal page area on {i+1}"
assert reference_page is not None
# References may begin on a fresh eighth page after exactly seven body pages.
# Count body content, rather than requiring the reference heading to share page 7.
reference_prefix = pdf[reference_page-1].get_text().split("References",1)[0].strip()
last_body_page = reference_page if reference_prefix else reference_page-1
assert last_body_page <= 7, f"Main content exceeds seven pages: last body page {last_body_page}"
assert len(pdf) <= 9
report = {
    "status": "pass",
    "kind": "manuscript_QA_not_experimental_evidence",
    "source_files": len(SECTIONS),
    "labels": len(labels),
    "citations": len(cites),
    "synthetic_algebra_cases": cases,
    "properties": ["fixed_hull", "affine_exposure", "endpoint_ratio",
                   "independent_scale_error_box", "constant_CI_energy_ranking"],
    "pdf_pages": len(pdf),
    "references_start_page": reference_page,
    "last_body_page": last_body_page,
    "pending_experiments": ["E1","E2","E3","E4"],
    "visual_review_required": True
}
out = ROOT/"tmp/pdfs/manuscript_check.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
