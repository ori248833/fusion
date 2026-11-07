"""Check recording/recovery and timestamp association using synthetic events."""

import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from my_launch.frame_recording import (
    create_run_directory, FIELDS, FrameRecorder, read_header_stamp, write_json,
    write_text_atomic,
)
from my_launch.frame_report import (
    generate_report, load_run, PeriodicReports, recover_runs, summarize,
)


def header(sec=100, nanosec=0, endian='<'):
    """Make the leading CDR1 bytes used by all monitored message types."""
    return (b'\x00\x01\x00\x00' if endian == '<' else b'\x00\x00\x00\x00') + struct.pack(
        endian + 'iI', sec, nanosec)


class RecordingTests(unittest.TestCase):
    """Use no ROS installation, sensor data, or inferred performance results."""

    def test_stamp_precision_endianness_and_invalid_encoding(self):
        self.assertEqual(read_header_stamp(header(123, 999999999)), (123, 999999999, 'ok'))
        self.assertEqual(read_header_stamp(header(0, 0, '>')), (0, 0, 'ok'))
        self.assertEqual(read_header_stamp(header(-2, 11, '>')), (-2, 11, 'ok'))
        self.assertEqual(read_header_stamp(header(nanosec=1000000000))[2], 'invalid_nanosec')
        self.assertEqual(read_header_stamp(b'123')[2], 'truncated')
        self.assertEqual(read_header_stamp(b'\x00\x07' + bytes(10))[2], 'unsupported_cdr')

    def test_complete_run_drains_queue_and_preserves_nanoseconds(self):
        with tempfile.TemporaryDirectory() as root:
            run = create_run_directory(root)
            self.assertNotEqual(run, create_run_directory(root))
            recorder = FrameRecorder(run, {'topics': {'yolo': '</script><script>bad()'}})
            origin = recorder.started_ns
            stamp = header(1788888888, 999999999)
            # The output arrives well after the input's one-second window.
            recorder.record('camera_raw', stamp, 1, received_ns=origin + 1_000_000_000)
            recorder.record('yolo', stamp, 2, received_ns=origin + 2_500_000_000)
            recorder.record('yolo', stamp, 3, received_ns=origin + 2_600_000_000)
            recorder.record('fusion', header(nanosec=1000000000), 4)
            metadata = recorder.close()
            self.assertEqual(metadata['state'], 'finished')
            self.assertEqual(recorder.close(), metadata)
            data = load_run(run)
            self.assertEqual(data['streams']['yolo'][0][1], '1788888888999999999')
            summary = summarize(data)
            links = summary['stamp_links']['camera_raw->yolo']
            self.assertEqual(links['observed_on_both_sides'], 1)
            self.assertEqual(links['upstream_stamps_not_observed_downstream'], 0)
            self.assertEqual(links['ambiguous_stamps'], 1)
            self.assertEqual(summary['streams']['fusion']['invalid_headers'], 1)
            page = generate_report(run).read_text(encoding='utf-8')
            self.assertNotIn('__RUN_DATA__', page)
            self.assertNotIn('__METRICS_JS__', page)
            self.assertNotIn('</script><script>bad()', page)
            payload = page.split('<script id="run-data" type="application/json">')[1]
            payload = payload.split('</script>')[0]
            self.assertEqual(
                json.loads(payload)['metadata']['topics']['yolo'], '</script><script>bad()')

    def test_recovery_of_unfinished_run_with_partial_tail(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            write_json(run / 'run.json', {'state': 'recording', 'started_system': 'test'})
            with (run / 'frames.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                writer.writerow(('lidar_raw', 1, 3000000000, 5, 6, 0, 0, 'ok'))
                handle.write('lidar_raw,2,4000000000')
            data = load_run(run)
            self.assertTrue(data['recovered'])
            self.assertEqual(data['duration'], 3.0)
            self.assertEqual(data['skipped_csv_rows'], 1)
            self.assertTrue(generate_report(run).exists())

    def test_full_queue_is_accounted_for_and_never_blocks_producer(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {}, capacity=1)
            entered, release = threading.Event(), threading.Event()
            original = recorder._writer

            class PausedWriter:
                def writerow(self, row):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('Test writer release missing')
                    original.writerow(row)

            recorder._writer = PausedWriter()
            try:
                recorder.record('camera_raw', header(), 1)
                self.assertTrue(entered.wait(2))
                recorder.record('camera_raw', header(), 2)
                recorder.record('camera_raw', header(), 3)
                self.assertEqual(recorder.dropped['camera_raw'], 1)
            finally:
                release.set()
                metadata = recorder.close()
            self.assertEqual(metadata['state'], 'incomplete')
            self.assertEqual(len(load_run(root)['streams']['camera_raw']), 2)

    def test_writer_failure_is_exposed_and_close_does_not_deadlock(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {}, capacity=1)
            failed = threading.Event()

            class FailingWriter:
                def writerow(self, row):
                    failed.set()
                    raise OSError('synthetic full disk')

            recorder._writer = FailingWriter()
            try:
                recorder.record('lidar_raw', header(), 1)
                self.assertTrue(failed.wait(2))
            finally:
                metadata = recorder.close()
            self.assertEqual(metadata['state'], 'incomplete')
            self.assertIn('synthetic full disk', metadata['writer_error'])

    def test_empty_run_and_stamp_regression(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {})
            recorder.close()
            self.assertTrue(generate_report(root).exists())
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {})
            recorder.record('lidar_raw', header(10), 1)
            recorder.record('lidar_raw', header(9), 2)
            recorder.close()
            summary = summarize(load_run(root))
            self.assertEqual(summary['streams']['lidar_raw']['source_stamp_regressions'], 1)

    def test_live_report_reads_only_the_committed_csv_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {}, sync_interval=30)
            initial = recorder.snapshot()
            try:
                recorder.record('lidar_raw', header(), 1)
                deadline = time.monotonic() + 3
                while not recorder._written_counts['lidar_raw'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(recorder._written_counts['lidar_raw'], 1)
                recorder._file.flush()  # A visible row is not yet a committed checkpoint.
                preview = load_run(root, metadata=initial, checkpoint_only=True)
                self.assertEqual(preview['streams']['lidar_raw'], [])
                self.assertTrue(preview['preview'])
                self.assertFalse(preview['recovered'])
                self.assertEqual(len(load_run(root)['streams']['lidar_raw']), 1)
            finally:
                recorder.close()

    def test_atomic_sync_failure_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'report.html'
            write_text_atomic(path, 'previous complete report')
            with patch('my_launch.frame_recording.os.fsync', side_effect=OSError('sync failed')):
                with self.assertRaises(OSError):
                    write_text_atomic(path, 'partial new report')
            self.assertEqual(path.read_text(encoding='utf-8'), 'previous complete report')
            self.assertEqual(list(Path(root).glob('*.tmp')), [])

    def test_truncated_checkpoint_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = FrameRecorder(Path(root), {})
            recorder.record('lidar_raw', header(), 1)
            recorder.close()
            path = Path(root) / 'frames.csv'
            path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
            data = load_run(root)
            self.assertEqual(data['checkpoint_shortfall']['lidar_raw'], 1)
            self.assertGreater(data['checkpoint_bytes_missing'], 0)
            self.assertEqual(summarize(data)['checkpoint_shortfall']['lidar_raw'], 1)

    def test_recovery_does_not_touch_a_live_run(self):
        with tempfile.TemporaryDirectory() as root:
            run = create_run_directory(root)
            recorder = FrameRecorder(run, {})
            try:
                recovered, errors = recover_runs(root)
                self.assertEqual((recovered, errors), ([], []))
                self.assertEqual(json.loads((run / 'run.json').read_text())['state'], 'recording')
            finally:
                recorder.close()

    def test_periodic_report_generation_runs_in_a_different_process(self):
        with tempfile.TemporaryDirectory() as root:
            run = create_run_directory(root)
            recorder = FrameRecorder(run, {})
            reports = PeriodicReports(recorder, interval=.1)
            try:
                self.assertNotEqual(reports._process.pid, os.getpid())
                self.assertTrue(reports._process.is_alive())
                recorder.record('lidar_raw', header(), 1)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    summary = json.loads((run / 'summary.json').read_text('utf-8'))
                    if summary['streams']['lidar_raw']['recorded_frames'] == 1:
                        break
                    time.sleep(.02)
                self.assertEqual(summary['streams']['lidar_raw']['recorded_frames'], 1)
                self.assertIsNone(reports.error)
            finally:
                reports.close()
                recorder.close()
            self.assertFalse(reports._process.is_alive())
            self.assertIsNone(reports.error)

    def test_hard_kill_preserves_periodic_report_and_recovers_without_close(self):
        # A process kill checks absence of finally/close; it is NOT a power-cut test.
        program = '''
from pathlib import Path
import struct
import sys
import time
from my_launch.frame_recording import FrameRecorder
from my_launch.frame_report import PeriodicReports
recorder = FrameRecorder(Path(sys.argv[1]), {}, sync_interval=.05)
reports = PeriodicReports(recorder, interval=.1)
index = 0
while True:
    stamp = b'\\x00\\x01\\x00\\x00' + struct.pack('<iI', 100, index)
    recorder.record('lidar_raw', stamp, index)
    index += 1
    time.sleep(.01)
'''
        with tempfile.TemporaryDirectory() as root:
            run = create_run_directory(root)
            environment = os.environ.copy()
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            process = subprocess.Popen(
                [sys.executable, '-u', '-c', program, str(run)], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            payload = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        page = (run / 'report.html').read_text(encoding='utf-8')
                        raw = page.split('<script id="run-data" type="application/json">')[1]
                        payload = json.loads(raw.split('</script>')[0])
                        if len(payload['streams']['lidar_raw']) >= 3:
                            break
                    except (FileNotFoundError, ValueError, IndexError):
                        pass
                    time.sleep(.02)
                self.assertIsNotNone(payload)
                self.assertGreaterEqual(len(payload['streams']['lidar_raw']), 3)
            finally:
                if process.poll() is None:
                    process.kill()
                _, stderr = process.communicate(timeout=5)
            self.assertEqual(stderr, b'')
            metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['state'], 'recording')
            self.assertNotIn('duration_ns', metadata)
            committed = metadata['checkpoint']['written_counts']['lidar_raw']
            self.assertGreaterEqual(len(load_run(run)['streams']['lidar_raw']), committed)
            original_csv = (run / 'frames.csv').read_bytes()
            replacement = FrameRecorder(create_run_directory(root), {})
            reports = PeriodicReports(replacement, interval=.1)
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if load_run(run)['metadata']['state'] == 'interrupted':
                        break
                    time.sleep(.02)
                self.assertEqual(reports.recovery_errors, [])
                self.assertEqual(load_run(run)['metadata']['state'], 'interrupted')
            finally:
                reports.close()
                replacement.close()
            self.assertEqual((run / 'frames.csv').read_bytes(), original_csv)
            self.assertEqual(load_run(run)['metadata']['state'], 'interrupted')
            self.assertTrue(load_run(run)['recovered'])
            self.assertEqual(recover_runs(root), ([], []))


if __name__ == '__main__':
    unittest.main()
