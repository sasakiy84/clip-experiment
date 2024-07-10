import csv
from pathlib import Path
from common.path import RESULT_DIR

RESULT_CSV = RESULT_DIR / "combined_result_sim.csv"

# result dir にある csv ファイルを結合する
result_files = list(RESULT_DIR.glob("sim_*.csv"))
result_csv_writer = csv.writer(RESULT_CSV.open("w"))

# ファイルのヘッダーが同じであることを確認
header = None

for result_file in result_files:
    with result_file.open() as f:
        reader = csv.reader(f)

        # header が全て同じであることを確認
        if header is None:
            header = next(reader)
            result_csv_writer.writerow(header)
        else:
            if header != next(reader):
                raise ValueError(f"Header mismatch: {result_file}")
        
        for row in reader:
            result_csv_writer.writerow(row)
