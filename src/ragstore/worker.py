"""Background ingestion worker.

Single in-process asyncio loop: claim the oldest queued job, process it, mark it
done (with chunk count) or failed (with the error string). Fail loud — errors are
recorded on the job, never swallowed.
"""

from __future__ import annotations

import asyncio
import logging

from ragstore.service import RagService

logger = logging.getLogger("ragstore.worker")


class IngestionWorker:
    def __init__(self, service: RagService, poll_interval: float = 0.2) -> None:
        self._service = service
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="ragstore-ingestion-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            processed = await self.run_once()
            if not processed:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
                except TimeoutError:
                    pass

    async def run_once(self) -> bool:
        """Claim and process a single job. Returns True if a job was handled."""
        job = await self._service.sqlite.claim_next_job()
        if job is None:
            return False
        try:
            chunks = await self._service.process_job(job)
            await self._service.sqlite.finish_job(job["id"], "done", chunks=chunks)
        except Exception as exc:  # noqa: BLE001 — record on the job, fail loud per-job
            logger.exception("job %s failed", job["id"])
            await self._service.sqlite.finish_job(job["id"], "failed", error=repr(exc))
        return True
