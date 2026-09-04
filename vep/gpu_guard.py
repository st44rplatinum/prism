"""Thermal watchdog for long GPU jobs.

    python -m vep.gpu_guard                    # guard the precompute job
    python -m vep.gpu_guard --dry-run          # monitor and log, act on nothing

Polls the GPU temperature and, when it climbs too far, suspends the target
process until it cools - duty-cycling the job rather than abandoning it. If the
temperature keeps rising past a hard limit, the process is killed instead.

Killing is a reasonable last resort *specifically because* the precompute job is
checkpointed per gene: at worst one gene's work is lost and `python -m
vep.precompute` picks up exactly where it stopped.

Thresholds default to this card's own reported limits, read from nvidia-smi:
a GTX 1050 Ti reports slowdown at 99C and shutdown at 102C, so acting at 83/90
leaves the driver's own protection as a backstop that should never be reached.

The one genuinely dangerous failure mode is leaving the target suspended, which
would look exactly like a hang. Every exit path resumes it - normal exit,
Ctrl+C, and unhandled exceptions alike.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = os.name == "nt"

# Cooperative pause file. The guard writes a deadline; a cooperating job checks
# it and sleeps until then.
#
# This is the default because suspending a process from outside is not safe to
# rely on here: on Windows, `taskkill` and shell `timeout` terminate via
# TerminateProcess, which no signal handler can intercept, so a guard killed
# mid-pause leaves its target frozen forever - indistinguishable from a hang.
# A deadline in a file cannot have that failure mode: if the guard dies, the
# deadline simply passes and the job carries on by itself.
PAUSE_FILE = Path("artifacts/cache/thermal_pause.json")


def request_pause(seconds: float, temp: int, path: Path = PAUSE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"until": time.time() + seconds, "temp": temp}), encoding="utf-8"
    )


def clear_pause(path: Path = PAUSE_FILE) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def pause_remaining(path: Path = PAUSE_FILE) -> float:
    """Seconds left on an active pause request, 0 if none or expired."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0.0
    return max(0.0, float(blob.get("until", 0)) - time.time())


def wait_if_paused(path: Path = PAUSE_FILE, poll: float = 0.5, on_wait=None) -> float:
    """Block while a thermal pause is in force. Returns seconds actually waited.

    Called by long GPU jobs between batches. Re-reads each poll so the guard can
    extend or lift the pause, and any stale deadline expires on its own.
    """
    waited = 0.0
    while True:
        left = pause_remaining(path)
        if left <= 0:
            return waited
        if on_wait is not None and waited == 0.0:
            on_wait(left)
        time.sleep(min(poll, left))
        waited += poll


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def gpu_stats() -> dict | None:
    """Current temperature, utilisation and memory, or None if unreadable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.hw_slowdown",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "temp": int(parts[0]),
            "util": int(parts[1]),
            "mem": int(parts[2]),
            "hw_slowdown": parts[3] if len(parts) > 3 else "unknown",
        }
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def reported_limits() -> dict:
    """Shutdown/slowdown thresholds the card advertises, for context in the log."""
    limits: dict[str, int] = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip().rstrip(" C")
            if not value.isdigit():
                continue
            k = key.strip().lower()
            if "shutdown" in k:
                limits["shutdown"] = int(value)
            elif "slowdown" in k:
                limits["slowdown"] = int(value)
            elif "target" in k:
                limits["target"] = int(value)
    except subprocess.SubprocessError:
        pass
    return limits


# ---------------------------------------------------------------------------
# process control
# ---------------------------------------------------------------------------
def find_pids(pattern: str, exe: str = "python") -> list[int]:
    """PIDs of `exe` processes whose command line contains `pattern`.

    The executable filter is not cosmetic. Matching on command line alone also
    catches the shells and wrappers that merely *mention* the script - a
    `nohup python -m vep.precompute` launch matches nohup.exe too, and a bash
    heredoc containing the text matches bash.exe. Suspending those does nothing
    useful, and killing a wrapper leaves the real GPU process running.
    """
    if IS_WINDOWS:
        script = (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.Name -like '*{exe}*' }} | "
            f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | "
            "Where-Object { $_.CommandLine -notlike '*gpu_guard*' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=30,
            )
            return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        except (subprocess.SubprocessError, ValueError):
            return []
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10
        )
        pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return []
    keep = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/comm") as fh:
                if exe in fh.read():
                    keep.append(pid)
        except OSError:
            continue
    return keep


def process_alive(pid: int) -> bool:
    if IS_WINDOWS:
        PROCESS_QUERY_LIMITED = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _nt(fn: str, pid: int) -> bool:
    PROCESS_ALL_ACCESS = 0x1F0FFF
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not h:
        return False
    try:
        return getattr(ctypes.windll.ntdll, fn)(h) == 0
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def suspend(pid: int) -> bool:
    """Freeze the process. CUDA work already queued finishes; no new work is issued."""
    if IS_WINDOWS:
        return _nt("NtSuspendProcess", pid)
    try:
        os.kill(pid, signal.SIGSTOP)
        return True
    except OSError:
        return False


def resume(pid: int) -> bool:
    if IS_WINDOWS:
        return _nt("NtResumeProcess", pid)
    try:
        os.kill(pid, signal.SIGCONT)
        return True
    except OSError:
        return False


def kill(pid: int) -> bool:
    if IS_WINDOWS:
        return subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)], capture_output=True
        ).returncode == 0
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------
@dataclass
class Guard:
    pids: list[int]
    throttle: int
    resume_at: int
    kill_at: int
    interval: float
    duty: float
    dry_run: bool = False
    mode: str = "pause"
    suspended: bool = False
    peak: int = 0
    throttle_events: int = 0
    seconds_suspended: float = 0.0
    log: list[str] = field(default_factory=list)

    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.append(line)

    def resume_all(self) -> None:
        """Idempotent: safe to call from any exit path, suspended or not."""
        if self.mode == "pause":
            clear_pause()
            return
        if not self.suspended:
            return
        for pid in self.pids:
            resume(pid)
        self.suspended = False
        self.say("resumed (guard exiting)")

    def _suspend_all(self) -> None:
        for pid in self.pids:
            suspend(pid)
        self.suspended = True

    def _resume_all(self) -> None:
        for pid in self.pids:
            resume(pid)
        self.suspended = False

    def run(self) -> int:
        atexit.register(self.resume_all)
        try:
            return self._loop()
        finally:
            self.resume_all()

    def _loop(self) -> int:
        while True:
            self.pids = [p for p in self.pids if process_alive(p)]
            if not self.pids:
                print()
                self.say("target process is gone; guard exiting")
                return 0

            stats = gpu_stats()
            if stats is None:
                print()
                self.say("could not read nvidia-smi; retrying")
                time.sleep(self.interval)
                continue

            temp = stats["temp"]
            self.peak = max(self.peak, temp)

            if temp >= self.kill_at:
                print()
                self.say(
                    f"{temp}C >= kill threshold {self.kill_at}C - terminating "
                    f"{self.pids}. Work is checkpointed per gene; resume with "
                    f"`python -m vep.precompute`."
                )
                if not self.dry_run:
                    self._resume_all()          # never kill a suspended process
                    for pid in self.pids:
                        kill(pid)
                return 1

            if temp >= self.throttle:
                self.throttle_events += 1
                print()
                self.say(f"{temp}C >= {self.throttle}C - pausing {self.duty:.0f}s "
                         f"({self.mode})")
                if not self.dry_run and self.mode == "pause":
                    # Deadline in a file: the job pauses itself, and a guard
                    # that dies leaves nothing stuck.
                    request_pause(self.duty, temp)
                    slept = 0.0
                    while slept < self.duty and not _stopping:
                        time.sleep(min(1.0, self.duty - slept))
                        slept += 1.0
                    self.seconds_suspended += slept
                    after = gpu_stats()
                    if after and after["temp"] < self.resume_at:
                        clear_pause()
                        self.say(f"cooled to {after['temp']}C - pause lifted")
                    else:
                        self.say(f"still {after['temp'] if after else '?'}C")
                elif not self.dry_run:
                    # Suspension is always bracketed by the sleep in this one
                    # branch and released before the next iteration. Holding it
                    # across iterations was how an earlier version left a
                    # process frozen when the guard itself was killed - and a
                    # frozen job is indistinguishable from a hung one.
                    self._suspend_all()
                    try:
                        slept = 0.0
                        while slept < self.duty and not _stopping:
                            time.sleep(min(1.0, self.duty - slept))
                            slept += 1.0
                        self.seconds_suspended += slept
                    finally:
                        self._resume_all()
                    after = gpu_stats()
                    self.say(f"resumed at {after['temp'] if after else '?'}C")
                if _stopping:
                    return 130
                continue

            print(
                f"\r  {temp}C  util {stats['util']:3d}%  mem {stats['mem']:5d} MiB  "
                f"peak {self.peak}C  throttles {self.throttle_events}   ",
                end="",
                flush=True,
            )
            time.sleep(self.interval)
            if _stopping:
                print()
                self.say("stop requested")
                return 130


_stopping = False


def _request_stop(signum, frame):
    """Ask the loop to unwind so the finally-blocks resume the target.

    Registered for SIGTERM as well as SIGINT: Python runs neither atexit nor
    finally on a default SIGTERM, so without this a `taskkill` (or a shell
    `timeout`) on the guard would leave the job suspended indefinitely.
    """
    global _stopping
    _stopping = True


def unfreeze(pattern: str, exe: str) -> int:
    """Recovery: resume anything this guard may have left suspended.

    The residual risk is a guard killed with SIGKILL / `taskkill /F`, which no
    handler can intercept. This is the way out of that state.
    """
    pids = find_pids(pattern, exe)
    if not pids:
        print(f"no process matching {pattern!r}")
        return 1
    for pid in pids:
        ok = resume(pid)
        print(f"  resumed PID {pid}: {'ok' if ok else 'FAILED'}")
    return 0


def main() -> int:
    limits = reported_limits()
    ap = argparse.ArgumentParser(description="Suspend or kill a GPU job when it runs hot.")
    ap.add_argument("--pattern", default="vep.precompute",
                    help="match the target by command line (default: vep.precompute)")
    ap.add_argument("--pid", type=int, action="append", help="target PID directly; repeatable")
    ap.add_argument("--exe", default="python",
                    help="only match processes whose executable name contains this "
                         "(default: python), so shell wrappers are not targeted")
    ap.add_argument("--throttle", type=int, default=83,
                    help="suspend at or above this temperature (default 83, the card's target)")
    ap.add_argument("--resume-at", type=int, default=78, help="resume below this (default 78)")
    ap.add_argument("--kill-at", type=int, default=90,
                    help="hard kill at or above this (default 90)")
    ap.add_argument("--interval", type=float, default=5.0, help="poll seconds (default 5)")
    ap.add_argument("--duty", type=float, default=20.0,
                    help="seconds to stay suspended per throttle event (default 20)")
    ap.add_argument("--mode", default="pause", choices=("pause", "suspend"),
                    help="pause: write a deadline the job honours itself (default, "
                         "self-healing). suspend: freeze the process externally - "
                         "works on any process, but a hard-killed guard can strand it")
    ap.add_argument("--dry-run", action="store_true", help="log decisions, change nothing")
    ap.add_argument("--unfreeze", action="store_true",
                    help="resume a target left suspended by a hard-killed guard, then exit")
    args = ap.parse_args()

    if args.unfreeze:
        return unfreeze(args.pattern, args.exe)

    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), _request_stop)
            except (ValueError, OSError):
                pass

    if not (args.resume_at < args.throttle < args.kill_at):
        raise SystemExit(
            f"thresholds must satisfy resume_at < throttle < kill_at "
            f"(got {args.resume_at} < {args.throttle} < {args.kill_at})"
        )

    stats = gpu_stats()
    if stats is None:
        raise SystemExit("cannot read nvidia-smi - is a NVIDIA GPU present?")

    pids = args.pid or find_pids(args.pattern, args.exe)
    if not pids:
        raise SystemExit(
            f"no process matching {args.pattern!r}. Start the job first, or pass --pid."
        )

    print(f"guarding PIDs {pids} (match: {args.pattern!r})")
    if limits:
        print(f"card reports: " + ", ".join(f"{k} {v}C" for k, v in sorted(limits.items())))
    print(f"throttle >= {args.throttle}C, resume < {args.resume_at}C, kill >= {args.kill_at}C")
    print(f"mode: {args.mode}")
    print(f"currently {stats['temp']}C, util {stats['util']}%"
          + ("   [DRY RUN - no action taken]" if args.dry_run else ""))
    print()

    guard = Guard(
        mode=args.mode,
        pids=pids,
        throttle=args.throttle,
        resume_at=args.resume_at,
        kill_at=args.kill_at,
        interval=args.interval,
        duty=args.duty,
        dry_run=args.dry_run,
    )
    try:
        code = guard.run()
    except KeyboardInterrupt:
        guard.say("interrupted")
        code = 130
    print(f"\npeak {guard.peak}C, {guard.throttle_events} throttle events, "
          f"{guard.seconds_suspended:.0f}s suspended")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
