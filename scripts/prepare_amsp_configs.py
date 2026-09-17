#!/usr/bin/env python3
"""Build explicit retrospective development configs, without scheduler runs.

All source timestamps are interpreted as UTC scenario coordinates. This is
not an assertion about the original logs' time zones or an actual deployment.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCES = {
    'frontera': ('data/filtered/filtered-frontera-rtx.log', '2019-12-05','2021-08-19',
                 [('2020-01-05','2020-09-01'),('2020-10-01','2021-01-01'),('2021-02-01','2021-08-01')]),
    'iw': ('data/new/filtered_iw_log.log','2022-12-31','2024-07-22',
           [('2023-02-01','2023-10-01'),('2023-11-01','2024-02-01'),('2024-03-01','2024-07-01')]),
}


def build(data_root, output_root):
    here=Path(__file__).resolve().parents[1]
    template=json.loads((here/'configs/cluster.template.json').read_text())
    public_path=here/'data/amsp/profiles.json'
    public=json.loads(public_path.read_text())
    data_root,output_root=Path(data_root),Path(output_root)
    (output_root/'configs').mkdir(parents=True,exist_ok=True)
    cohorts=output_root/'data/amsp/cohorts';cohorts.mkdir(parents=True,exist_ok=True)
    ci_path='data/texas_eia/ci.csv';ci_hash=sha(data_root/ci_path)
    produced=[]
    for name,(trace,begin,end,intervals) in SOURCES.items():
        trace_hash=sha(data_root/trace)
        rng=random.Random(20260917)
        rows=['episode_id,arrival_utc,split,budget_hours']
        for split,(left,right) in zip(('train','validation','test'),intervals):
            at=datetime.fromisoformat(left).replace(tzinfo=timezone.utc)
            stop=datetime.fromisoformat(right).replace(tzinfo=timezone.utc)-timedelta(days=21)
            index=0
            while at<stop:
                arrival=at+timedelta(hours=rng.randrange(24))
                rows.append(f'{name}-{split}-{index:04d},{arrival.isoformat().replace("+00:00","Z")},{split},192')
                at+=timedelta(days=3);index+=1
        cohort_rel=f'data/amsp/cohorts/{name}.csv'
        cohort_path=output_root/cohort_rel
        cohort_path.write_text('\n'.join(rows)+'\n')
        for model,workload in public['workloads'].items():
            c=deepcopy(template)
            c.update(purpose='development',panel=f'AMSP-{model}-{name}-C128',root='..')
            c['cluster'].update(name='AMSP-shaped-128-node-scenario',partition='constructed-homogeneous-nodes',nodes=128,
                allowed_nodes=[4,16,64,128],provenance='Declared 128-node contention scenario; published target profile has 8 A800 per node. Not a reconstruction of the historical source cluster.')
            c['trace'].update(path=trace,sha256=trace_hash,timezone='UTC',dst_fold=None,
                coverage_start_utc=begin+'T00:00:00Z',coverage_end_utc=end+'T00:00:00Z',
                coverage_attestation='Constructed finite arrival stream from retained records in this interval; collection completeness of the original cluster is not asserted.',
                provenance='Historical '+name+' resource-demand template. Naive timestamps assigned UTC scenario coordinates, paired with ERCOT UTC calendar. Original timezone and deployment association not asserted.',
                role='workload_template',node_multiplier=1,oversize_policy='error')
            c['workload']=deepcopy(workload)
            c['workload']['provenance']+=f' Input record SHA256: {sha(public_path)}.'
            c['ci'].update(path=ci_path,sha256=ci_hash,source='EIA ERCOT Published Hourly Data; see data/texas_eia/source.json',
                region='Texas ERCOT scenario',unit='gCO2/kWh',type='average_operational',
                availability_rule='Revised archive made available 24h after interval end as a declared scenario; not vintage data.',
                alignment_description='UTC scenario coordinates from naive source timestamps overlaid on ERCOT calendar; no claim of actual source timezone or A800 testbed location.')
            c['power'].update(reference_kw=1.0,workload_coefficient=None,
                provenance='1 kW is a normalization unit, not a server specification or measured power; unknown kappa absorbs absolute node power. Report ratios and node-hours.')
            c['splits']={split:[left+'T00:00:00Z',right+'T00:00:00Z'] for split,(left,right) in zip(('train','validation','test'),intervals)}
            c['cohort'].update(path=cohort_rel,sha256=sha(cohort_path),
                provenance='Predeclared every-third-day arrivals with seeded hour, seed 20260917; 21-day end guard. Same arrivals across profiles/methods; 192h field is replaced by trained budget grid during policy evaluation.')
            c['execution'].update(warmup_seconds=28*86400,initial_state_mode='empty_warmup')
            c['holdout_audit'].update(test_is_untouched=False,evaluation_design='retrospective_temporal',
                notes='Author reports prior train/validation use. New chronological splits fixed before this revision runs; old data are reused and no previously unseen test claim is made. Development until protocol/software are frozen.')
            path=output_root/f'configs/amsp-{name}-{model.lower()}.development.json'
            path.write_text(json.dumps(c,indent=2)+'\n');produced.append(str(path))
    return produced


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output-root',default=str(Path(__file__).resolve().parents[1]))
    args=parser.parse_args()
    print('\n'.join(build(args.data_root,args.output_root)))
