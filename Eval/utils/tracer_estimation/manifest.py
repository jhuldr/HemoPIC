import json
from pathlib import Path


### Write a short readable manifest for outputs
def write_manifest(run_dir: Path, payload: dict):
    out_path = run_dir / "manifest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    return out_path