"""Chronologically cross-fitted wait distributions; inference is dependency-free JSON."""

from collections import defaultdict
import json
import math
from pathlib import Path
import struct
import time

from .common import ContractError, digest, integer, iso, json_text, load_json, require, timestamp
from .features import FEATURE_VERSION, QueueFeatures, quantile
from .runner import write_manifest


def queue_contract(manifest):
    config = manifest["resolved_config"]
    return {"trace": {k:v for k,v in config["trace"].items() if k not in {"path", "provenance"}},
            "execution": config["execution"], "cluster": config["cluster"]}


def export_regressor(model):
    trees = []
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                      "value": tree.value[:, 0, 0].tolist()})
    return {"initial": float(model.init_.constant_.ravel()[0]), "learning_rate": model.learning_rate,
            "trees": trees, "features": model.n_features_in_, "input_dtype": "float32"}


def predict_regressor(model, vector):
    require(len(vector) == model["features"], "Wait model feature count differs")
    # sklearn's tree predictor converts to float32 before threshold comparisons.
    vector = [struct.unpack("f", struct.pack("f", float(v)))[0] for v in vector]
    value = model["initial"]
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            node = tree["left"][node] if vector[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
        value += model["learning_rate"] * tree["value"][node]
    require(math.isfinite(value), "Nonfinite wait prediction")
    return value


class WaitPredictor:
    def __init__(self, artifact, version="in-memory"):
        require(artifact["kind"] == "wait_distribution_v1", "Unsupported wait model format")
        schema = artifact["feature_schema"]
        require(schema["version"] == FEATURE_VERSION, "Unsupported wait feature schema")
        self.features = QueueFeatures(schema["lags"], schema["calendar_timezone"])
        require(self.features.names == schema["names"], "Wait model feature ordering differs")
        self.artifact, self.version = artifact, version
        self.supported_nodes = [int(n) for n in artifact["residual_atoms"]]

    @classmethod
    def load(cls, path):
        return cls(load_json(path), digest(path))

    def atoms_from_features(self, vector, nodes):
        require(str(nodes) in self.artifact["residual_atoms"], f"No wait residuals for scale {nodes}")
        mean_log = predict_regressor(self.artifact["regressor"], vector)
        try:
            atoms = [max(0.0, math.expm1(mean_log + residual)) for residual in self.artifact["residual_atoms"][str(nodes)]]
        except OverflowError:
            raise ContractError("Wait model overflow; audit fit/residuals rather than silently clipping") from None
        require(all(math.isfinite(v) for v in atoms), "Nonfinite wait atom")
        return atoms

    def check_inputs(self, manifest):
        require(self.artifact["trace_sha256"] == manifest["asset_sha256"]["trace"], "Wait model trace differs")
        require(self.artifact["queue_contract"] == queue_contract(manifest), "Wait model queue interpretation differs")

    def atoms(self, history, at, nodes, requested_seconds):
        return self.atoms_from_features(self.features.request(history, at, nodes, requested_seconds), nodes)


def chronological_folds(rows, folds):
    """Group simultaneous requests and purge labels not observed before each fold."""
    times = sorted({timestamp(r["arrival_utc"]) for r in rows})
    folds = integer(folds, "folds", 2)
    require(len(times) >= folds + 1, "Need at least folds+1 distinct training snapshot times")
    boundaries = [i * len(times) // (folds + 1) for i in range(folds + 2)]
    for fold in range(1, folds + 1):
        begin_index, end_index = boundaries[fold], boundaries[fold + 1]
        begin = times[begin_index]
        selected_times = set(times[begin_index:end_index])
        training = [i for i, r in enumerate(rows) if timestamp(r["arrival_utc"]) < begin and
                    timestamp(r["label_observed_at_utc"]) < begin]
        validation = [i for i, r in enumerate(rows) if timestamp(r["arrival_utc"]) in selected_times]
        require(training and validation, "Empty chronological fold after label-overlap purge; provide more probe history")
        yield training, validation, begin


def fit_wait_model(probe_dir, output, trees=256, depth=4, learning_rate=.05, folds=4, atoms=32, seed=11):
    try:
        import numpy as np
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        raise ContractError('Wait fitting needs scikit-learn: "$CARBON_PYTHON" -m pip install -r requirements-p2.txt') from None
    probe_dir, output = Path(probe_dir), Path(output)
    meta = load_json(probe_dir / "manifest.json")
    require(meta["status"] == "complete" and meta["kind"] == "wait_probes", "Probe dataset is incomplete")
    require(meta["probe_split"] == "train", "Wait model must fit training probes only")
    require(digest(probe_dir / "probes.jsonl") == meta["probes_sha256"], "Probe dataset hash mismatch")
    require(not output.exists(), f"Output already exists: {output}")
    trees, depth, atoms = integer(trees, "trees"), integer(depth, "depth"), integer(atoms, "atoms")
    require(math.isfinite(learning_rate) and learning_rate > 0, "Invalid wait fitting settings")
    with (probe_dir / "probes.jsonl").open(encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream]
    rows = []
    for row in all_rows:
        require(row["split"] == "train", "Nontraining row in wait fitting input")
        if row["censored"]:
            require(row["wait_hours"] is None, "Censored probe has a fabricated complete wait")
            continue
        arrival, observed = timestamp(row["arrival_utc"]), timestamp(row["label_observed_at_utc"])
        require(arrival <= observed < timestamp(meta["label_boundary_utc"]), "Wait label crosses its training boundary")
        require(math.isclose((observed - arrival).total_seconds()/3600, row["wait_hours"], abs_tol=1e-9), "Wait label and timestamp disagree")
        rows.append(row)
    require(rows, "No completed training wait labels")
    x = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.log1p(np.asarray([r["wait_hours"] for r in rows], dtype=float))
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite training data")
    require(x.shape[1] == len(meta["feature_schema"]["names"]), "Probe features disagree with schema")
    options = dict(n_estimators=int(trees), max_depth=int(depth), learning_rate=float(learning_rate),
                   loss="squared_error", random_state=int(seed), n_iter_no_change=None)
    residuals, audits = defaultdict(list), []
    begun = time.monotonic()
    for train, test, begin in chronological_folds(rows, folds):
        model = GradientBoostingRegressor(**options).fit(x[train], y[train])
        predicted = model.predict(x[test])
        for index, prediction in zip(test, predicted):
            residuals[rows[index]["nodes"]].append(float(y[index] - prediction))
        audits.append({"first_validation_arrival_utc": iso(begin), "train_rows": len(train),
                       "validation_rows": len(test),
                       "last_training_label_observed_utc": max((rows[i]["label_observed_at_utc"] for i in train), key=timestamp)})
    allowed = meta["resolved_config"]["cluster"]["allowed_nodes"]
    require(set(residuals) == set(allowed), "Every action needs out-of-fold wait residuals; provide more probes")
    model = GradientBoostingRegressor(**options).fit(x, y)
    exported = export_regressor(model)
    # Verify portable inference against the actual fitted library model before saving.
    check = x[np.linspace(0, len(x)-1, min(128, len(x)), dtype=int)]
    require(np.allclose([predict_regressor(exported, row) for row in check], model.predict(check), rtol=1e-10, atol=1e-12),
            "Portable GBT export differs from fitted model")
    artifact = {"kind": "wait_distribution_v1", "feature_schema": meta["feature_schema"],
                "regressor": exported,
                "residual_atoms": {str(n): [quantile(residuals[n], (j + .5)/atoms) for j in range(atoms)] for n in allowed},
                "probes_sha256": meta["probes_sha256"], "train_label_boundary_utc": meta["label_boundary_utc"],
                "fit_options": options, "folds": audits, "residual_counts": {str(n): len(residuals[n]) for n in allowed},
                "training_rows": len(rows), "censored_training_rows": len(all_rows)-len(rows),
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                "fit_seconds": time.monotonic()-begun,
                "purpose": meta["purpose"], "cluster": meta["resolved_config"]["cluster"],
                "queue_contract": queue_contract(meta),
                "trace_sha256": meta["asset_sha256"]["trace"],
                "calibration_claim": "empirical scale-specific OOF residual atoms, not a probability guarantee"}
    output.mkdir(parents=True)
    (output / "model.json").write_text(json_text(artifact), encoding="utf-8")
    write_manifest(output / "manifest.json", {"status": "complete", "kind": "wait_model", "model_sha256": digest(output / "model.json"),
                                               "training_rows": len(rows), "fit_seconds": artifact["fit_seconds"],
                                               "folds": audits, "purpose": meta["purpose"]})
    return artifact
