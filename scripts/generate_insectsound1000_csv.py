import os
import csv
import re

# Path to the directory containing the audio files
DATA_DIR = "data/raw/insectsound1000"

# Output CSV file
OUTPUT_CSV = "insectsound1000_metadata.csv"

# Regex to parse the filename
FILENAME_REGEX = re.compile(
    r"(?P<date>\d{7,8}-\d{1,2}-\d{1,2})_(?P<species>[A-Za-z_]+)_(?P<sample_id>\d+)_s(?P<segment>\d+)_ch(?P<channel>\d)\.wav"
)

def parse_filename(filename):
    match = FILENAME_REGEX.match(filename)
    if not match:
        return None
    return match.groupdict()

def main():
    rows = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith(".wav"):
                continue
            parsed = parse_filename(file)
            if not parsed:
                continue
            row = {
                "filepath": os.path.join(root, file),
                "recording_date": parsed["date"],
                "species": parsed["species"].replace("_", " "),
                "sample_id": parsed["sample_id"],
                "segment": parsed["segment"],
                "channel": parsed["channel"],
            }
            rows.append(row)

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["filepath", "recording_date", "species", "sample_id", "segment", "channel"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
