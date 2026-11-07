"""Build or recover a self-contained frame-rate report without ROS or a server."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import queue
import statistics
import time

from .frame_recording import RunLease, STREAMS, write_json, write_text_atomic


def _csv_lines(handle, byte_limit):
    consumed = 0
    for line in handle:
        consumed += len(line)
        if byte_limit is not None and consumed > byte_limit:
            break
        yield line.decode('utf-8', errors='replace')


def load_run(run_dir, metadata=None, checkpoint_only=False):
    """Load recorded metadata, preserving nanosecond stamps as integer strings."""
    run_dir = Path(run_dir)
    if metadata is None:
        metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    checkpoint = metadata.get('checkpoint', {})
    limit = checkpoint.get('csv_bytes') if checkpoint_only else None
    if checkpoint_only and limit is None:
        raise ValueError('A live report requires a committed CSV checkpoint')
    streams = {key: [] for key, _ in STREAMS}
    skipped = 0
    with (run_dir / 'frames.csv').open('rb') as handle:
        for row in csv.DictReader(_csv_lines(handle, limit)):
            try:
                if not row.get('header_status'):
                    raise ValueError('Incomplete CSV row')
                int(row['receive_system_ns'])
                int(row['receive_ros_ns'])
                key = row['stream']
                elapsed = int(row['elapsed_ns']) / 1e9
                source = None
                if row['header_status'] == 'ok':
                    source = str(int(row['source_sec']) * 1_000_000_000
                                 + int(row['source_nanosec']))
                event = [elapsed, source, int(row['sequence']), row['header_status']]
                if key not in streams or elapsed < 0:
                    raise ValueError('Unknown stream or negative elapsed time')
                streams[key].append(event)
            except (ValueError, KeyError, TypeError):
                # A killed process can leave a partial final CSV row.
                skipped += 1
    for events in streams.values():
        events.sort(key=lambda event: event[0])
    observed_end = max((events[-1][0] for events in streams.values() if events), default=0)
    duration = max(observed_end, metadata.get('duration_ns', 0) / 1e9,
                   checkpoint.get('elapsed_ns', 0) / 1e9)
    preview = checkpoint_only and metadata.get('state') == 'recording'
    shortfall = {key: max(0, count - len(streams[key])) for key, count
                 in checkpoint.get('written_counts', {}).items() if key in streams}
    return {
        'metadata': metadata, 'duration': duration, 'streams': streams,
        'labels': dict(STREAMS), 'skipped_csv_rows': skipped,
        'recovered': 'duration_ns' not in metadata and not preview,
        'preview': preview,
        'checkpoint_shortfall': shortfall,
        'checkpoint_bytes_missing': max(
            0, checkpoint.get('csv_bytes', 0) - (run_dir / 'frames.csv').stat().st_size),
    }


def summarize(data):
    """Summarize observed continuity; unmatched stamps are never confirmed drops."""
    summary = {
        'measurement': 'independent_subscriber_receive_time',
        'state': data['metadata'].get('state'),
        'duration_seconds': data['duration'],
        'recovered': data['recovered'],
        'preview': data['preview'],
        'checkpoint': data['metadata'].get('checkpoint'),
        'checkpoint_shortfall': data['checkpoint_shortfall'],
        'checkpoint_bytes_missing': data['checkpoint_bytes_missing'],
        'skipped_csv_rows': data['skipped_csv_rows'],
        'recorder_dropped_rows': data['metadata'].get('recorder_dropped_rows', {}),
        'streams': {}, 'stamp_links': {},
    }
    for key, events in data['streams'].items():
        gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
        stamps = [int(event[1]) for event in events if event[1] is not None]
        summary['streams'][key] = {
            'recorded_frames': len(events),
            'average_hz_over_run': len(events) / data['duration'] if data['duration'] else 0,
            'first_receive_seconds': events[0][0] if events else None,
            'last_receive_seconds': events[-1][0] if events else None,
            'max_interarrival_seconds': max(gaps) if gaps else None,
            'median_interarrival_seconds': statistics.median(gaps) if gaps else None,
            'interarrival_average_hz': len(gaps) / sum(gaps) if gaps and sum(gaps) > 0 else None,
            'callback_count_at_checkpoint': data['metadata'].get('received_counts', {}).get(key),
            'writer_dropped_rows': data['metadata'].get('recorder_dropped_rows', {}).get(key, 0),
            'invalid_headers': sum(event[1] is None for event in events),
            'repeated_source_stamps': len(stamps) - len(set(stamps)),
            'source_stamp_regressions': sum(b < a for a, b in zip(stamps, stamps[1:])),
        }
    for upstream, downstream in (
            ('lidar_raw', 'lidar_detection'), ('camera_raw', 'yolo'),
            ('lidar_detection', 'fusion')):
        source = Counter(event[1] for event in data['streams'][upstream]
                         if event[1] is not None)
        target = Counter(event[1] for event in data['streams'][downstream]
                         if event[1] is not None)
        summary['stamp_links'][f'{upstream}->{downstream}'] = {
            'upstream_unique_stamps': len(source),
            'observed_on_both_sides': len(source.keys() & target.keys()),
            'upstream_stamps_not_observed_downstream': len(source.keys() - target.keys()),
            'ambiguous_stamps': sum(source[s] > 1 or target[s] > 1
                                    for s in source.keys() & target.keys()),
            'interpretation': 'Candidate links only. Unobserved does not prove a dropped frame.',
        }
    return summary


def generate_report(run_dir, metadata=None, checkpoint_only=False):
    """Generate one HTML file that works offline when opened directly."""
    run_dir = Path(run_dir)
    lease = RunLease(run_dir, filename='.report.lock')
    if not lease.acquire():
        raise BlockingIOError(f'Another process is updating the report: {run_dir}')
    try:
        return _generate_report(run_dir, metadata, checkpoint_only)
    finally:
        lease.close()


def _generate_report(run_dir, metadata, checkpoint_only):
    data = load_run(run_dir, metadata=metadata, checkpoint_only=checkpoint_only)
    data['summary'] = summarize(data)
    write_json(run_dir / 'summary.json', data['summary'])
    template = Path(__file__).with_name('frame_report.html').read_text(encoding='utf-8')
    metrics = Path(__file__).with_name('frame_metrics.js').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    # Escape HTML delimiters in user-controlled paths/topics inside the script.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    report = run_dir / 'report.html'
    write_text_atomic(
        report, template.replace('__METRICS_JS__', metrics).replace('__RUN_DATA__', payload))
    return report


def recover_runs(root, exclude=None):
    """Recover unclosed runs under one output root, skipping any live OS lock."""
    root = Path(root).expanduser().resolve()
    exclude = Path(exclude).resolve() if exclude is not None else None
    recovered, errors = [], []
    for folder in root.iterdir():
        if folder == exclude or not folder.is_dir() or folder.is_symlink():
            continue
        metadata_path = folder / 'run.json'
        if not metadata_path.is_file() or not (folder / 'frames.csv').is_file():
            continue
        lease = RunLease(folder)
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (metadata.get('state') not in ('recording', 'incomplete')
                    or 'duration_ns' in metadata
                    or metadata.get('measurement') != 'independent_subscriber_receive_time'):
                continue
            if not lease.acquire():
                continue
            # Recheck after taking ownership: the old recorder may just have closed.
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (metadata.get('state') not in ('recording', 'incomplete')
                    or 'duration_ns' in metadata):
                continue
            metadata.update({
                'state': 'interrupted',
                'stop_reason': 'unclean_exit_detected_on_next_start',
                'recovered_system': datetime.now().astimezone().isoformat(),
            })
            generate_report(folder, metadata=metadata)
            write_json(metadata_path, metadata)
            recovered.append(str(folder))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{folder}: {type(error).__name__}: {error}')
        finally:
            lease.close()
    return recovered, errors


def _report_worker(run_dir, interval, stop, status):
    """Keep CSV parsing, JSON encoding and HTML generation outside the collector."""
    run_dir = Path(run_dir)
    parent = multiprocessing.parent_process()
    while not stop.is_set() and (parent is None or parent.is_alive()):
        error, recovery_errors = None, []
        try:
            recovered, recovery_errors = recover_runs(run_dir.parent, exclude=run_dir)
            for folder in recovered:
                print(f'Recovered interrupted frame recording: {folder}', flush=True)
            generate_report(run_dir, checkpoint_only=True)
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
        try:
            status.put_nowait((error, recovery_errors))
        except queue.Full:
            try:
                status.get_nowait()
                status.put_nowait((error, recovery_errors))
            except (queue.Empty, queue.Full):
                pass
        # A hard-killed collector cannot signal stop. Watch its OS process sentinel.
        deadline = time.monotonic() + interval
        while not stop.is_set() and time.monotonic() < deadline:
            if parent is not None and not parent.is_alive():
                return
            stop.wait(min(.1, max(0, deadline - time.monotonic())))


class PeriodicReports:
    """Build reports in a spawned process, sharing no collector Python GIL."""

    def __init__(self, recorder, interval=5.0):
        self.interval = max(0.1, float(interval))
        self._error = None
        self._recovery_errors = []
        self._closed = False
        context = multiprocessing.get_context('spawn')
        self._stop = context.Event()
        self._status = context.Queue(maxsize=8)
        # Even a run shorter than one report interval has an openable HTML file.
        generate_report(recorder.run_dir, metadata=recorder.snapshot(), checkpoint_only=True)
        self._process = context.Process(
            target=_report_worker,
            args=(str(recorder.run_dir), self.interval, self._stop, self._status),
            name='frame-report-writer', daemon=True)
        self._process.start()

    def _refresh_status(self, check_exit=True):
        if self._closed:
            return
        while True:
            try:
                self._error, self._recovery_errors = self._status.get_nowait()
            except queue.Empty:
                break
            except (OSError, EOFError) as error:
                self._error = f'Report status channel failed: {error}'
                break
        if check_exit and self._process.exitcode is not None:
            self._error = f'Report process exited unexpectedly: {self._process.exitcode}'

    @property
    def error(self):
        """Return report errors without parsing any recording data in this process."""
        self._refresh_status()
        return self._error

    @property
    def recovery_errors(self):
        """Return the latest recovery status from the report process."""
        self._refresh_status()
        return self._recovery_errors

    def close(self):
        """Finish a current atomic report update before the final shutdown report."""
        if self._closed:
            return
        self._stop.set()
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            error = 'Report shutdown timed out; previous atomic report is preserved'
        else:
            error = None
        self._refresh_status(check_exit=False)
        if error is not None:
            self._error = error
        self._closed = True
        self._status.close()


def main():
    """Rebuild reports from complete or interrupted run folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    print(generate_report(args.run_directory))


if __name__ == '__main__':
    main()
