"""Durable, bounded metadata recording without any ROS dependency."""

import csv
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import queue
import struct
import tempfile
import threading
import time


STREAMS = (
    ('lidar_raw', '原始雷达'),
    ('camera_raw', '原始相机'),
    ('lidar_detection', '雷达检测输出'),
    ('yolo', 'YOLO 输出'),
    ('fusion', '融合 Map 输出'),
)
FIELDS = (
    'stream', 'sequence', 'elapsed_ns', 'receive_system_ns', 'receive_ros_ns',
    'source_sec', 'source_nanosec', 'header_status',
)


def read_header_stamp(data):
    """Read the leading Header stamp from plain CDR1, without decoding payloads.

    All five configured message types start with std_msgs/Header. Reject other
    encodings explicitly instead of silently inventing timestamps. A zero stamp
    is valid (e.g. the first frame in a simulated run).
    """
    if len(data) < 12:
        return '', '', 'truncated'
    encoding = bytes(data[:2])
    if encoding not in (b'\x00\x00', b'\x00\x01'):
        return '', '', 'unsupported_cdr'
    sec, nanosec = struct.unpack_from(
        '<iI' if encoding == b'\x00\x01' else '>iI', data, 4)
    if nanosec >= 1_000_000_000:
        return '', '', 'invalid_nanosec'
    return sec, nanosec, 'ok'


def sync_directory(path):
    """Persist directory entries on the target Linux filesystem."""
    if os.name == 'posix':
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_text_atomic(path, content):
    """Sync a temporary file, replace its destination, then sync the directory."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, value):
    """Atomically persist JSON, retaining the previous version until replacement."""
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2))


def _mkdir_durable(path):
    if not path.exists():
        _mkdir_durable(path.parent)
        path.mkdir(exist_ok=True)
        sync_directory(path.parent)


class RunLease:
    """Use an OS-released lock to keep recovery away from a running recorder."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / '.recorder.lock'
        self.handle = None

    def acquire(self):
        """Return false if another process currently owns the run directory."""
        handle = self.path.open('a+b', buffering=0)
        try:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == 'nt':
                import msvcrt
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b'\0')
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                raise OSError('Run locking is unsupported on this platform')
        except OSError as error:
            handle.close()
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        self.handle = handle
        return True

    def close(self):
        """Release the lock; a crash also releases it through the operating system."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def create_run_directory(root):
    """Create a unique directory named using the host's local system time."""
    root = Path(root).expanduser().resolve()
    _mkdir_durable(root)
    name = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S_%f')
    for suffix in range(10000):
        path = root / (name if suffix == 0 else f'{name}_{suffix}')
        try:
            path.mkdir()
            sync_directory(root)
            return path
        except FileExistsError:
            continue
    raise FileExistsError('Cannot allocate a unique run directory')


class FrameRecorder:
    """Write bounded queued rows and checkpoint them to storage periodically."""

    def __init__(self, run_dir, metadata, capacity=20000, sync_interval=0.1):
        self.run_dir = Path(run_dir)
        self.sync_interval = max(0.05, float(sync_interval))
        self._lease = RunLease(run_dir)
        if not self._lease.acquire():
            raise RuntimeError(f'Run is already being recorded: {run_dir}')
        self.metadata = dict(metadata)
        self._metadata_lock = threading.Lock()
        self.started_ns = time.monotonic_ns()
        self.metadata.update({
            'schema_version': 1,
            'started_system': datetime.now().astimezone().isoformat(),
            'started_system_ns': time.time_ns(),
            'state': 'recording',
            'measurement': 'independent_subscriber_receive_time',
            'durability': {
                'method': 'flush + fsync CSV, atomic fsynced checkpoint JSON',
                'sync_interval_seconds': self.sync_interval,
                'directory_fsync': os.name == 'posix',
                'loss_window': 'Target interval only; scheduling, queue backlog and storage '
                               'latency can enlarge the uncommitted tail.',
            },
        })
        self.counts = {key: 0 for key, _ in STREAMS}
        self.dropped = {key: 0 for key, _ in STREAMS}
        self.error = None
        self._closed = False
        self._queue = queue.Queue(maxsize=max(1, capacity))
        self._stop = threading.Event()
        self._written_counts = {key: 0 for key, _ in STREAMS}
        self._last_frame_elapsed_ns = 0
        self._file = None
        try:
            self._file = (self.run_dir / 'frames.csv').open(
                'x', newline='', encoding='utf-8')
            self._writer = csv.writer(self._file)
            self._writer.writerow(FIELDS)
            self._checkpoint()
        except Exception:
            if self._file is not None:
                self._file.close()
            self._lease.close()
            raise
        self._thread = threading.Thread(
            target=self._write_loop, name='frame-metadata-writer', daemon=True)
        self._thread.start()

    def record(self, stream, data, receive_ros_ns, received_ns=None,
               receive_system_ns=None):
        """Enqueue only numbers; never retain a point cloud or image payload."""
        if self._closed:
            return
        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        system_ns = time.time_ns() if receive_system_ns is None else receive_system_ns
        self.counts[stream] += 1
        sec, nanosec, status = read_header_stamp(data)
        row = (stream, self.counts[stream], received_ns - self.started_ns,
               system_ns, receive_ros_ns, sec, nanosec, status)
        if self.error is not None:
            self.dropped[stream] += 1
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped[stream] += 1

    def snapshot(self):
        """Copy the last committed checkpoint without waiting for disk operations."""
        with self._metadata_lock:
            return json.loads(json.dumps(self.metadata))

    def _checkpoint(self):
        checkpoint_time = time.monotonic_ns()
        self._file.flush()
        os.fsync(self._file.fileno())
        metadata = self.snapshot()
        metadata.update({
            'received_counts': self.counts.copy(),
            'recorder_dropped_rows': self.dropped.copy(),
            'writer_error': self.error,
            'checkpoint': {
                'elapsed_ns': checkpoint_time - self.started_ns,
                'system_time': datetime.now().astimezone().isoformat(),
                'csv_bytes': os.fstat(self._file.fileno()).st_size,
                'written_counts': self._written_counts.copy(),
                'last_frame_elapsed_ns': self._last_frame_elapsed_ns,
                'queued_rows_at_checkpoint': self._queue.qsize(),
            },
        })
        # Commit ordering: CSV fsync -> checkpoint fsync/replace -> directory fsync.
        write_json(self.run_dir / 'run.json', metadata)
        with self._metadata_lock:
            self.metadata = metadata

    def _write_loop(self):
        last_sync = time.monotonic()
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=min(0.1, self.sync_interval))
                    self._writer.writerow(row)
                    self._written_counts[row[0]] += 1
                    self._last_frame_elapsed_ns = max(self._last_frame_elapsed_ns, row[2])
                except queue.Empty:
                    pass
                if time.monotonic() - last_sync >= self.sync_interval:
                    self._checkpoint()
                    last_sync = time.monotonic()
            self._checkpoint()
        except Exception as error:  # Report disk failure; never block perception.
            self.error = f'{type(error).__name__}: {error}'
            metadata = self.snapshot()
            metadata.update({
                'state': 'incomplete', 'writer_error': self.error,
                'received_counts': self.counts.copy(),
                'recorder_dropped_rows': self.dropped.copy(),
            })
            with self._metadata_lock:
                self.metadata = metadata
            try:
                write_json(self.run_dir / 'run.json', metadata)
            except OSError:
                pass  # Storage failure may also prevent saving the error itself.
        finally:
            try:
                self._file.close()
            except Exception as error:
                self.error = f'{type(error).__name__}: {error}'

    def close(self, reason='shutdown'):
        """Drain pending metadata and mark whether the recording is complete."""
        if self._closed:
            return self.metadata
        self._closed = True
        duration_ns = time.monotonic_ns() - self.started_ns
        self._stop.set()
        self._thread.join()
        self.metadata.update({
            'state': 'incomplete' if (
                self.error or any(self.dropped.values())
                or reason not in ('shutdown', 'interrupt_or_context_shutdown')) else 'finished',
            'stop_reason': reason,
            'stopped_system': datetime.now().astimezone().isoformat(),
            'duration_ns': duration_ns,
            'received_counts': self.counts,
            'recorder_dropped_rows': self.dropped,
            'writer_error': self.error,
        })
        try:
            write_json(self.run_dir / 'run.json', self.metadata)
        finally:
            self._lease.close()
        return self.metadata
