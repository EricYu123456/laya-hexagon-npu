"""Reuse measured Conv offsets only for exactly identical prepared probes.

This runs on the calibrated Pi after sourcing npu/env.sh. It checks the current
backend identity but does not compile or run an accelerator graph. An affine-only
change outside the Convs can qualify; a different bucket/probe cannot.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import onnx

from npu import calibrate_conv_offsets as cal
from npu.fidelity_runtime import qnn_backend_fingerprint, qnn_backend_identity


def current_runtime_fingerprint():
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    return qnn_backend_fingerprint(ort.__version__, importlib.metadata.version("onnxruntime-qnn"),
                                   qnn.get_qnn_htp_path(), htp_arch=68, require_auxiliary=True)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source(manifest, source, label):
    source = Path(source if source else manifest["source_path"]).resolve()
    if cal.sha256(source) != manifest["source_sha256"]:
        raise ValueError(f"{label} source checksum differs from its prepared probe manifest")
    model = onnx.load(source)
    if any(item.name.startswith("__htp_offset_") for item in model.graph.initializer):
        raise ValueError(f"{label} source already has Conv offset corrections")
    if cal.conv_records(model) != manifest["records"]:
        raise ValueError(f"{label} source Conv records differ from its prepared probe manifest")
    return source


def _validate_measured_arrays(donor, manifest, result):
    records = manifest["records"]
    ids = [record["id"] for record in records]
    shape_ids = manifest["shape_validation_ids"]
    if (not ids or len(ids) != len(set(ids)) or not shape_ids
            or len(shape_ids) != len(set(shape_ids)) or not set(shape_ids) <= set(ids)):
        raise ValueError("Invalid or duplicate Conv/spatial validation IDs")
    if result.get("all_shape_validations_passed") is not True:
        raise ValueError("Donor spatial validation did not pass")
    with np.load(donor / "correction_biases.npz", allow_pickle=False) as saved:
        corrections = {name: saved[name] for name in saved.files}
    if set(corrections) != set(ids):
        raise ValueError("Donor correction arrays do not cover exactly the Conv IDs")
    expected_raw_keys = {f"{kind}_{key}_small_y" for kind in ("cpu", "htp") for key in ids}
    expected_raw_keys.update(f"{kind}_{key}_original_y" for kind in ("cpu", "htp") for key in shape_ids)
    with np.load(donor / "zero_outputs.npz", allow_pickle=False) as raw:
        if set(raw.files) != expected_raw_keys:
            raise ValueError("Donor raw measurements do not cover exactly the prepared probes")
        validation, summaries = [], []
        for record in records:
            key, channels = record["id"], record["output_channels"]

            def pair(role, shape):
                values = [raw[f"{kind}_{key}_{role}_y"] for kind in ("cpu", "htp")]
                if any(value.shape != tuple(shape) or value.dtype != np.float32
                       or not np.isfinite(value).all() for value in values):
                    raise ValueError(f"Invalid raw measurement shape/type/values: {key}/{role}")
                return values

            cpu, htp = pair("small", [1, channels, 1, 1])
            delta = (htp - cpu).reshape(-1).astype(np.float32)
            correction = corrections[key]
            if correction.dtype != np.float32 or not np.array_equal(correction, -delta):
                raise ValueError(f"Donor correction does not equal negative measured offset: {key}")
            summaries.append({"id": key, "node": record["node"],
                              "mean_abs_offset": float(np.abs(delta).mean()),
                              "max_abs_offset": float(np.abs(delta).max()),
                              "output_step": record["output_scale"]})
            if key in shape_ids:
                cpu, htp = pair("original", [1, channels, *record["input_shape"][2:]])
                maximum = float(np.abs((htp - cpu) - delta.reshape(1, -1, 1, 1)).max())
                tolerance = record["output_scale"] * .25 + 1e-7
                validation.append({"id": key, "max_abs_offset_difference": maximum,
                                   "tolerance": tolerance, "passed": bool(maximum <= tolerance)})
    if (validation != result.get("shape_validation") or not all(item["passed"] for item in validation)
            or summaries != result.get("per_conv")):
        raise ValueError("Donor result summaries/spatial checks disagree with raw measurements")


def reuse(*, from_dir, out, source=None, from_source=None):
    """Validate measured donor evidence, then bind it to identical recipient probes."""
    donor, recipient = Path(from_dir).resolve(), Path(out).resolve()
    if donor == recipient:
        raise ValueError("Donor and recipient probe directories must be separate")
    output_names = ("zero_outputs.npz", "correction_biases.npz", "calibration_results.json")
    if any((recipient / name).exists() for name in output_names):
        raise FileExistsError("Recipient already has measurements; prepare a fresh probe directory")
    donor_manifest_path, recipient_manifest_path = donor / "probe_manifest.json", recipient / "probe_manifest.json"
    donor_manifest, recipient_manifest = _read_json(donor_manifest_path), _read_json(recipient_manifest_path)
    if donor_manifest.get("format_version", 1) < 2 or recipient_manifest.get("format_version", 1) < 2:
        raise ValueError("Calibration reuse requires version-2 hashed probe manifests")
    donor_probe_hash, recipient_probe_hash = cal.sha256(donor / "zero_probes.onnx"), cal.sha256(recipient / "zero_probes.onnx")
    if (donor_probe_hash != donor_manifest["probe_sha256"]
            or recipient_probe_hash != recipient_manifest["probe_sha256"]
            or donor_probe_hash != recipient_probe_hash):
        raise ValueError("Donor and recipient must have byte-identical, checksum-verified probe graphs")
    for key in ("records", "shape_validation_ids", "calibration_inputs"):
        if donor_manifest[key] != recipient_manifest[key]:
            raise ValueError(f"Donor and recipient {key} differ; calibration reuse is forbidden")
    donor_source = _verify_source(donor_manifest, from_source, "Donor")
    recipient_source = _verify_source(recipient_manifest, source, "Recipient")
    result_path = donor / "calibration_results.json"
    result = _read_json(result_path)
    if "calibration_reuse" in result:
        raise ValueError("Use the original measured donor; chained calibration reuse is unsupported")
    if (result.get("source_sha256") != donor_manifest["source_sha256"]
            or result.get("probe_sha256") != donor_probe_hash
            or result.get("probe_manifest_sha256") != cal.sha256(donor_manifest_path)):
        raise ValueError("Donor result checksums do not match its source/probe/manifest")
    correction_hash = cal.sha256(donor / "correction_biases.npz")
    if correction_hash != result.get("correction_biases_sha256"):
        raise ValueError("Donor correction array checksum differs from the measured result")
    _validate_measured_arrays(donor, donor_manifest, result)
    runtime = result.get("runtime", {})
    required_fields = {"onnxruntime", "onnxruntime_qnn", "htp_arch", "backend_sha256", "stub_sha256", "skel_sha256"}
    if not required_fields.issubset(runtime) or runtime.get("cpu_ep_fallback") is not False:
        raise ValueError("Donor lacks a complete strict-HTP runtime identity")
    current = current_runtime_fingerprint()
    if current.get("cpu_ep_fallback") is not False or qnn_backend_identity(current) != qnn_backend_identity(runtime):
        raise ValueError("Current QNN runtime identity differs from the measured donor")
    result_hash, raw_hash = cal.sha256(result_path), cal.sha256(donor / "zero_outputs.npz")
    rebound = copy.deepcopy(result)
    rebound.update({"source_sha256": recipient_manifest["source_sha256"],
                    "probe_manifest_sha256": cal.sha256(recipient_manifest_path),
                    "calibration_reuse": {
                        "method": "identical prepared probe bytes and Conv records; raw measurements revalidated",
                        "donor_source": str(donor_source), "recipient_source": str(recipient_source),
                        "donor_source_sha256": donor_manifest["source_sha256"],
                        "donor_probe_sha256": donor_probe_hash,
                        "donor_probe_manifest_sha256": cal.sha256(donor_manifest_path),
                        "donor_calibration_results_sha256": result_hash,
                        "donor_zero_outputs_sha256": raw_hash,
                        "donor_correction_biases_sha256": correction_hash,
                        "verified_runtime_identity": qnn_backend_identity(current),
                    }})
    # No overwrite: partial artifacts after an I/O interruption remain visible.
    for name, expected_hash in (("zero_outputs.npz", raw_hash), ("correction_biases.npz", correction_hash)):
        with (donor / name).open("rb") as reader, (recipient / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if cal.sha256(recipient / name) != expected_hash:
            raise ValueError(f"Donor {name} changed while copying; recipient is incomplete")
    with (recipient / "calibration_results.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(rebound, indent=2, allow_nan=False) + "\n")
    return rebound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_dir", required=True, help="Original measured donor probe directory")
    parser.add_argument("--out", required=True, help="Fresh recipient prepared probe directory")
    parser.add_argument("--source", help="Recipient source graph after relocation; default: recipient manifest source_path")
    parser.add_argument("--from-source", help="Donor source graph after relocation; default: donor manifest source_path")
    result = reuse(**vars(parser.parse_args()))
    print(json.dumps({"reused_convs": len(result["per_conv"]), "source_sha256": result["source_sha256"],
                      "calibration_reuse": result["calibration_reuse"]}, indent=2))


if __name__ == "__main__":
    main()
