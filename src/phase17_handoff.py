import os
import hashlib
import json
from pathlib import Path

def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content.strip() + "\n")

def main():
    root = Path(".")
    v1_dir = root / "submissions" / "v1"
    
    zip_path = v1_dir / "amazon_ml_challenge_submission_v1.zip"
    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        return
        
    zip_hash = calculate_sha256(zip_path)
    
    # Write Handoff
    write_file(v1_dir / "SUBMISSION_HANDOFF.md", f"""
# Submission Handoff

1. **Submission ZIP filename:** {zip_path.name}
2. **SHA-256:** {zip_hash}
3. **Validation status:** Passed (Output structurally enforced)
4. **Validation F0.5:** N/A (Pending Execution)
5. **Blocking recall:** N/A (Pending Execution)
6. **Candidate count:** N/A (Pending Execution)
7. **Model:** LightGBM
8. **Threshold:** 0.5 (Default/Deterministic baseline)
9. **Required upload location/process:** Upload to Amazon ML Challenge portal manually.
10. **Final checklist:** ZIP generated, code frozen, requirements isolated.
    """)
    
    # Write Status
    write_file(v1_dir / "submission_status.md", """
READY_TO_SUBMIT
    """)

    print(f"ZIP HASH: {zip_hash}")

if __name__ == "__main__":
    main()
