import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent
data_dir = base_dir / "data" / "processed"
train_path = data_dir / "train_cleaned.json"

if train_path.exists():
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    null_count = 0
    for idx, item in enumerate(data):
        if item.get("id") is None or str(item.get("id")).strip() == "" or str(item.get("id")).lower() in ["null", "none"]:
            item["id"] = f"train_{idx + 1:05d}"
            null_count += 1

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Checked train_cleaned.json: Fixed {null_count} null IDs. Total samples: {len(data)}")

test_path = data_dir / "test_full.json"
if test_path.exists():
    with open(test_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)

    t_null_count = 0
    for idx, item in enumerate(t_data):
        if item.get("id") is None or str(item.get("id")).strip() == "" or str(item.get("id")).lower() in ["null", "none"]:
            item["id"] = f"test_{idx + 1:05d}"
            t_null_count += 1

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(t_data, f, ensure_ascii=False, indent=2)

    print(f"Checked test_full.json: Fixed {t_null_count} null IDs. Total samples: {len(t_data)}")
