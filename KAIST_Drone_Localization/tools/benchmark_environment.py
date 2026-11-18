"""Record the executable, configuration, input hashes, and process resources."""

from hashlib import sha256
import os
from pathlib import Path
import platform
import sys
from threading import active_count

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def code_digests():
    files = [ROOT / "main.py"]
    for folder in ("localization", "simulation", "tools"):
        files.extend((ROOT / folder).glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in sorted(files)}


def benchmark_environment(input_kind, input_label):
    inputs = {}
    if input_kind == "video_replay":
        path = Path(input_label)
        for member in (path, path.with_suffix(".csv"), path.with_suffix(".json")):
            if member.is_file():
                inputs[member.name] = dict(sha256=file_digest(member), bytes=member.stat().st_size)
    build = cv2.getBuildInformation().splitlines()
    parallel = next((line.split(":", 1)[1].strip() for line in build
                     if "Parallel framework:" in line), None)
    return dict(
        python=sys.version.split()[0], platform=platform.platform(),
        processor=os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or None,
        logical_cpu_count=os.cpu_count(), opencv=cv2.__version__,
        opencv_threads=cv2.getNumThreads(), opencv_parallel_framework=parallel,
        numpy=np.__version__, blas=np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas"),
        thread_environment={key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM")},
        input_files=inputs, code_sha256=code_digests(),
    )


def process_resources():
    """Sample resident process memory where the OS exposes it."""
    memory = system_cpu_times = None
    if sys.platform == "win32":
        memory, system_cpu_times = _windows_resources()
    elif Path("/proc/self/status").is_file():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                memory = int(line.split()[1]) * 1024
                break
    return dict(rss_bytes=memory, python_thread_count=active_count(), system_cpu_times=system_cpu_times)


def _windows_resources():
    from ctypes import POINTER, Structure, WinDLL, byref, c_size_t, sizeof
    from ctypes.wintypes import DWORD, FILETIME, HANDLE

    class MemoryCounters(Structure):
        _fields_ = [("cb", DWORD), ("PageFaultCount", DWORD)] + [
            (field, c_size_t) for field in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    kernel = WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = HANDLE
    memory = MemoryCounters(cb=sizeof(MemoryCounters))
    read_memory = kernel.K32GetProcessMemoryInfo
    read_memory.argtypes = [HANDLE, POINTER(MemoryCounters), DWORD]
    ok = read_memory(kernel.GetCurrentProcess(), byref(memory), sizeof(memory))
    idle, kernel_time, user = FILETIME(), FILETIME(), FILETIME()
    read_cpu = kernel.GetSystemTimes(byref(idle), byref(kernel_time), byref(user))
    times = tuple((time.dwHighDateTime << 32) + time.dwLowDateTime
                  for time in (idle, kernel_time, user)) if read_cpu else None
    return memory.WorkingSetSize if ok else None, times
