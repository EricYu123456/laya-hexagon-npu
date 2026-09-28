"""Combine independently verified buckets without copying large model files."""
import argparse
import json
import os
from pathlib import Path

from npu.fidelity_runtime import load_bucket_manifest, qnn_backend_identity, sha256_file


def merge_manifests(manifest_paths, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output manifest already exists")
    if not manifest_paths:
        raise ValueError("At least one source manifest is required")
    merged = None
    correction_backend = {}
    bucket_keys = {"buckets", "model_sha256", "htp_conv_offset_correction"}
    for source in manifest_paths:
        paths, metadata = load_bucket_manifest(source)
        policy = {key: value for key, value in metadata.items() if key not in bucket_keys}
        if merged is None:
            merged = {**policy, "buckets": {}, "model_sha256": {}}
        elif policy != {key: value for key, value in merged.items() if key not in bucket_keys}:
            raise ValueError("Source manifests have different checkpoint or inference policies")
        corrections = metadata.get("htp_conv_offset_correction", {})
        if set(corrections) - {str(bucket) for bucket in paths}:
            raise ValueError("Correction metadata contains unknown buckets")
        for bucket, path in paths.items():
            key = str(bucket)
            if key in merged["buckets"]:
                raise ValueError(f"Duplicate bucket {bucket}")
            digest = sha256_file(path)
            if metadata.get("model_sha256", {}).get(key) != digest:
                raise ValueError(f"Missing or mismatched model SHA256 for bucket {bucket}")
            if key in corrections:
                if corrections[key].get("output_sha256") != digest:
                    raise ValueError(f"Correction provenance has wrong output SHA256 for bucket {bucket}")
                for field, value in qnn_backend_identity(corrections[key]["runtime"]).items():
                    if field in correction_backend and correction_backend[field] != value:
                        raise ValueError(f"Corrected buckets require different backend {field}")
                    correction_backend[field] = value
                merged.setdefault("htp_conv_offset_correction", {})[key] = corrections[key]
            try:
                filename = os.path.relpath(path, output.parent)
            except ValueError:
                filename = str(path)
            merged["buckets"][key] = filename
            merged["model_sha256"][key] = digest
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    result = merge_manifests(args.manifests, args.output)
    print(json.dumps({"manifest": str(args.output), "buckets": result["buckets"]}, indent=2))


if __name__ == "__main__":
    main()
