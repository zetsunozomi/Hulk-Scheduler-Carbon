"""Wait-prediction diagnostics on independent replay probes, not recorded-job fidelity."""

from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean
import time

from .common import digest, require, timestamp
from .runner import provenance, write_manifest, write_record
from .waits import WaitPredictor


def atom_quantile(atoms, probability):
    """Inverse CDF of an equally weighted discrete predictive distribution."""
    require(atoms and 0 < probability <= 1, "Invalid atom quantile")
    return sorted(atoms)[max(0, math.ceil(probability * len(atoms)) - 1)]


def wait_scores(atoms, observed):
    require(atoms and math.isfinite(observed) and observed >= 0 and
            all(math.isfinite(v) and v >= 0 for v in atoms), "Invalid wait scores")
    ordered = sorted(atoms)
    n = len(atoms)
    median = atom_quantile(atoms, .5)
    mean_error = fmean(atoms) - observed
    # Half the pairwise absolute-difference expectation, in O(n log n).
    spread = sum((2*i-n+1)*value for i,value in enumerate(ordered)) / n**2
    return {"median_absolute_error_hours": abs(median-observed),
            "mean_error_hours": mean_error, "mean_squared_error_hours2": mean_error**2,
            "crps_hours": max(0.0, fmean(abs(v-observed) for v in atoms)-spread),
            "p90_coverage": float(observed <= atom_quantile(atoms, .9)),
            "central80_coverage": float(atom_quantile(atoms, .1) <= observed <= atom_quantile(atoms, .9)),
            "central80_width_hours": atom_quantile(atoms, .9)-atom_quantile(atoms, .1)}


def summarize_scores(records):
    observed = [r["scores"] for r in records if not r["censored"]]
    result = {"rows": len(records), "labeled_rows": len(observed), "censored_rows": len(records)-len(observed)}
    if not observed:
        return {**result, "metrics": None}
    scores = {key: fmean(r[key] for r in observed) for key in observed[0]}
    absolute = [r['median_absolute_error_hours'] for r in observed]
    scores['mae_of_median_hours'] = scores.pop('median_absolute_error_hours')
    scores['p50_absolute_error_of_median_hours'] = atom_quantile(absolute,.5)
    scores['p90_absolute_error_of_median_hours'] = atom_quantile(absolute,.9)
    scores["rmse_of_mean_hours"] = math.sqrt(scores.pop("mean_squared_error_hours2"))
    return {**result, "metrics": scores}


def pairwise_rank_scores(records,allowed_nodes,request_seconds):
    """Compare node counts only at the same snapshot and requested duration.

    A censored node pair is unknown; true ties are counted separately; predicted
    ties receive half credit. The predicted mean ranks the wait distribution.
    """
    groups = defaultdict(dict)
    for row in records:
        key = row['snapshot_id'],row['requested_seconds']
        require(row['nodes'] not in groups[key],'Duplicate probe action in one snapshot/request-length group')
        groups[key][row['nodes']] = row
    require(groups,'Empty ranking cohort')
    expected = set(allowed_nodes)
    snapshots = {key[0] for key in groups}
    require(set(groups)=={(s,length) for s in snapshots for length in request_seconds},'Missing probe request length in snapshot')
    counters = {'pairs':0,'censored_pairs':0,'observed_ties':0,'ordered_pairs':0,'predicted_ties':0,'correct_order_credit':0.}
    group_records = []
    for (snapshot,length),actions in sorted(groups.items()):
        require(set(actions)==expected,'Ranking requires every declared node count in each snapshot/request-length group')
        result = dict(counters)
        for a,b in combinations(sorted(actions),2):
            left,right = actions[a],actions[b]
            require(left['arrival_utc']==right['arrival_utc'],'Ranking snapshot arrivals differ')
            result['pairs'] += 1
            if left['censored'] or right['censored']:
                result['censored_pairs'] += 1
                continue
            difference = left['wait_hours']-right['wait_hours']
            if difference==0:
                result['observed_ties'] += 1
                continue
            result['ordered_pairs'] += 1
            predicted = fmean(left['predicted_atoms_hours'])-fmean(right['predicted_atoms_hours'])
            if predicted==0:
                result['predicted_ties'] += 1
                result['correct_order_credit'] += .5
            else:
                result['correct_order_credit'] += float(difference*predicted>0)
        result.update(snapshot_id=snapshot,requested_seconds=length,
                      accuracy=result['correct_order_credit']/result['ordered_pairs'] if result['ordered_pairs'] else None)
        group_records.append(result)
    totals = {key:sum(group[key] for group in group_records) for key in counters}
    usable = [g['accuracy'] for g in group_records if g['accuracy'] is not None]
    return {**totals,'snapshot_length_groups':len(groups),'groups_with_ordered_pairs':len(usable),
            'pair_weighted_accuracy':totals['correct_order_credit']/totals['ordered_pairs'] if totals['ordered_pairs'] else None,
            'mean_group_accuracy':fmean(usable) if usable else None,'groups':group_records,
            'scope':'same-snapshot same-request-length node pairs; mean predicted wait; ties half credit; simulator probe truth, not historical alternatives'}


def evaluate_waits(probe_directory, predictor_path, output):
    source, output = Path(probe_directory), Path(output)
    meta = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    require(meta["kind"] == "wait_probes" and meta["status"] == "complete", "Incomplete probe evaluation input")
    require(meta["probe_split"] in {"validation", "test"}, "Wait evaluation requires a held-out split")
    require(digest(source/"probes.jsonl") == meta["probes_sha256"], "Probe dataset hash mismatch")
    predictor = WaitPredictor.load(predictor_path)
    predictor.check_inputs(meta)
    require(predictor.artifact["feature_schema"] == meta["feature_schema"], "Evaluation feature schema differs")
    require(timestamp(predictor.artifact["train_label_boundary_utc"]) <= timestamp(meta["probe_start_utc"]),
            "Wait evaluation overlaps predictor training")
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    manifest = {"kind": "wait_evaluation", "status": "running", "split": meta["probe_split"],
                "software": provenance(Path(__file__).resolve().parents[2]),
                "purpose": meta["purpose"], "probes_sha256": meta["probes_sha256"], "predictor_sha256": predictor.version,
                "queue_regimes": "queued if current pending nodes > 0; otherwise busy if available fraction < .5; otherwise light",
                "coverage_scope": "completed probe labels only; censored probes remain counted, no calibration guarantee",
                "uncertainty": "descriptive only; calendar-block uncertainty is not computed here"}
    write_manifest(output/"manifest.json", manifest)
    groups, rows = defaultdict(list), []
    names = meta["feature_schema"]["names"]
    available = names.index("lag_0.available_fraction")
    pending = names.index("lag_0.pending.node_fraction")
    try:
        with (source/"probes.jsonl").open(encoding="utf-8") as incoming, (output/"predictions.jsonl").open("x", encoding="utf-8") as outgoing:
            for row in map(json.loads, incoming):
                require(row["split"] == meta["probe_split"], "Mixed probe splits")
                vector = row["features"]
                regime = "queued" if vector[pending] > 0 else "busy" if vector[available] < .5 else "light"
                begun = time.perf_counter()
                atoms = predictor.atoms_from_features(vector, row["nodes"])
                inference_seconds = time.perf_counter()-begun
                if row["censored"]:
                    require(row["wait_hours"] is None, "Censored probe has a fabricated wait")
                    scores = None
                else:
                    require(timestamp(row["arrival_utc"]) <= timestamp(row["label_observed_at_utc"]) < timestamp(meta["label_boundary_utc"]),
                            "Evaluation wait crosses split boundary")
                    require(math.isclose((timestamp(row['label_observed_at_utc'])-timestamp(row['arrival_utc'])).total_seconds()/3600,
                                         row['wait_hours'],rel_tol=1e-9,abs_tol=1e-9),'Wait label and timestamp disagree')
                    scores = wait_scores(atoms, row["wait_hours"])
                record = {k:v for k,v in row.items() if k != "features"}
                record.update(queue_regime=regime, predicted_atoms_hours=atoms, scores=scores,
                              predictor_inference_seconds=inference_seconds)
                write_record(outgoing, record)
                rows.append(record)
                for category, key in (("nodes", str(row["nodes"])), ("requested_seconds", str(row["requested_seconds"])),
                                      ("queue_regime", regime), ("nodes_queue_regime", f"{row['nodes']}:{regime}")):
                    groups[(category,key)].append(record)
        report = {"overall": summarize_scores(rows), "groups": {}}
        report['pairwise_rank'] = pairwise_rank_scores(rows,meta['resolved_config']['cluster']['allowed_nodes'],meta['request_seconds'])
        report['predictor_inference'] = {'calls':len(rows),'total_seconds':sum(r['predictor_inference_seconds'] for r in rows),
                                         'mean_seconds':fmean(r['predictor_inference_seconds'] for r in rows),
                                         'scope':'CPU wall time for portable GBT/residual atoms; excludes file IO, replay and feature construction; not energy'}
        for (category,key), records in sorted(groups.items()):
            report["groups"].setdefault(category,{})[key] = summarize_scores(records)
        write_manifest(output/"metrics.json", report)
        manifest.update(status="complete", rows=len(rows), metrics_sha256=digest(output/"metrics.json"),
                        predictions_sha256=digest(output/"predictions.jsonl"))
    except Exception as exc:
        manifest.update(status="failed", failure={"error_type": type(exc).__name__, "message": str(exc)})
        write_manifest(output/"failure.json", manifest["failure"])
        raise
    finally:
        write_manifest(output/"manifest.json", manifest)
    return manifest
