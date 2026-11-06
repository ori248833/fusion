"""Build clearly labelled synthetic data for offline UI verification only."""

import argparse
import csv
from datetime import datetime

from my_launch.frame_recording import create_run_directory, FIELDS, write_json
from my_launch.frame_report import generate_report


def main():
    """Create a ten-second artificial run, including one deliberately quiet span."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root')
    args = parser.parse_args()
    run_dir = create_run_directory(args.output_root)
    records = []
    base = 1788888888000000000
    for index in range(100):
        acquired = index / 10
        stamp = base + index * 100000000
        sec, nano = divmod(stamp, 1000000000)
        records.append(('lidar_raw', acquired + .02, sec, nano))
        if not 4 <= acquired < 4.7:
            records.append(('lidar_detection', acquired + .10, sec, nano))
            records.append(('fusion', acquired + .11, sec, nano))
    for index in range(300):
        acquired = index / 30
        stamp = base + index * 33333333
        sec, nano = divmod(stamp, 1000000000)
        records.append(('camera_raw', acquired + .02, sec, nano))
        if index % 3 != 1 and acquired < 9.8:
            records.append(('yolo', acquired + .16, sec, nano))
    counts = {}
    with (run_dir / 'frames.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for key, elapsed, sec, nano in sorted(records, key=lambda row: row[1]):
            counts[key] = counts.get(key, 0) + 1
            writer.writerow((key, counts[key], round(elapsed * 1e9),
                             base + round(elapsed * 1e9),
                             base + round(elapsed * 1e9), sec, nano, 'ok'))
    write_json(run_dir / 'run.json', {
        'schema_version': 1, 'state': 'finished', 'synthetic': True,
        'started_system': datetime.now().astimezone().isoformat(),
        'duration_ns': 10000000000, 'received_counts': counts,
        'recorder_dropped_rows': {}, 'parameters': {'yolo_max_fps': 20.0},
    })
    print(generate_report(run_dir))


if __name__ == '__main__':
    main()
