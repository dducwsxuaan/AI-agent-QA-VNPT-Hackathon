import json
import os

INPUT_FOLDER = "test_data"
OUTPUT_FOLDER = "test_data_splitted"
INPUT_FILE = "test.json"
BATCH_SIZE = 40

input_path = os.path.join(INPUT_FOLDER, INPUT_FILE)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

batches = [
    data[i:i + BATCH_SIZE]
    for i in range(0, len(data), BATCH_SIZE)
]

# Lưu từng batch sang folder output
for idx, batch in enumerate(batches):
    output_file = os.path.join(OUTPUT_FOLDER, f"batch_{idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)

print(f"✅ Đã chia {len(batches)} batch vào folder '{OUTPUT_FOLDER}'")