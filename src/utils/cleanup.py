import os
from pathlib import Path

def purge_legacy_files():
    base_dir = Path(__file__).resolve().parent.parent.parent
    scripts_dir = base_dir / "scripts"
    
    valid_scripts = {
        "01_prepare_data.py",
        "02_distill_teacher.py",
        "03_export_sft_datasets.py",
        "04_train_student.py",
        "05_evaluate.py",
        "06_demo_cli.py"
    }

    if scripts_dir.exists():
        for fpath in scripts_dir.glob("*.py"):
            if fpath.name not in valid_scripts:
                try:
                    fpath.unlink()
                    print(f"Purged legacy script: {fpath.name}")
                except Exception as e:
                    print(f"Could not delete {fpath.name}: {e}")

if __name__ == "__main__":
    purge_legacy_files()
