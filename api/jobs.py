"""Background jobs for masked-marginal saturation scans.

Masked-marginals cost one forward pass per residue - measured at roughly 7
positions/second on a GTX 1050 Ti, and slower still on long proteins where
every window runs the full 1022 tokens. A 2000-residue gene is therefore about
13 minutes, which is far past any sensible HTTP timeout.

So the saturation endpoint serves wt-marginals synchronously (one pass per
window, sub-second even for RYR1) and computes the stronger masked-marginal
matrix here in the background, promoting it into the store's cache when it
finishes. The front-end renders the fast matrix immediately and swaps.

This is an in-process registry: jobs die with the server. That is the right
weight for a single-user research tool - reach for a real queue only when the
API needs to survive a restart.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field

from api.store import Store


@dataclass
class Job:
    job_id: str
    gene: str
    scheme: str
    status: str = "queued"        # queued | running | done | error
    progress: float = 0.0
    detail: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "gene": self.gene,
                "scheme": self.scheme,
                "status": self.status,
                "progress": round(self.progress, 4),
                "detail": self.detail,
            }


class JobRegistry:
    def __init__(self, store: Store):
        self.store = store
        self._jobs: dict[str, Job] = {}
        self._by_target: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        # One worker at a time: the GPU is the bottleneck and two concurrent
        # scans would simply contend for it while doubling peak memory.
        self._gpu_lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def find_for(self, gene: str, scheme: str) -> Job | None:
        job_id = self._by_target.get((gene.upper(), scheme))
        return self._jobs.get(job_id) if job_id else None

    def submit(self, gene: str, scheme: str = "masked") -> Job:
        """Start a scan, or return the existing job for this target."""
        gene = gene.upper()
        with self._lock:
            existing = self.find_for(gene, scheme)
            if existing and existing.status in ("queued", "running"):
                return existing

            job = Job(job_id=uuid.uuid4().hex[:12], gene=gene, scheme=scheme)
            self._jobs[job.job_id] = job
            self._by_target[(gene, scheme)] = job.job_id

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job) -> None:
        try:
            with self._gpu_lock:
                job.update(status="running", progress=0.0)
                sequence = self.store.sequence(job.gene)
                if sequence is None:
                    job.update(status="error", detail=f"unknown gene {job.gene}")
                    return

                def on_progress(done: int, total: int) -> None:
                    job.update(progress=done / max(total, 1))

                backbone = self.store.backbone()
                log_probs = backbone.masked_marginals(
                    sequence, positions=None, batch_size=16, progress=on_progress
                )
                from vep.esm.backbone import llr_matrix

                matrix = llr_matrix(log_probs, sequence)
                self.store.put_saturation(job.gene, job.scheme, matrix)
                job.update(status="done", progress=1.0)
        except Exception as exc:  # surfaced to the client, not swallowed
            job.update(
                status="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()
