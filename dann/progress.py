"""Wall-clock phase reporting without tensor reads or CUDA synchronization."""
from __future__ import annotations

from time import perf_counter


class EpochProgress:
    """Report phase boundaries, first work, throughput, and estimated time left.

    Counts come from host-side loop bookkeeping. Timings include queueing and
    existing synchronization in the caller; reporting never waits on the GPU.
    """

    def __init__(self, epoch: int, total_epochs: int, interval_seconds: float) -> None:
        self.label = f'Epoch {epoch}/{total_epochs}'
        self.interval = interval_seconds
        self.details: dict = {}
        self.phase: str | None = None

    def set_phase(self, name: str, total: int = 0, *, unit: str = 'rows',
                  completed: int = 0) -> None:
        """Announce a new phase, keeping any resume offset out of throughput."""
        self.finish_phase()
        self.phase, self.total, self.unit = name, total, unit
        self.completed = self.initial = completed
        self.rows = 0
        self.started = self.last_report = perf_counter()
        self.reported_work = False
        print(f'{self.label} {name}: starting ({completed}/{total} {unit}).', flush=True)

    def update(self, completed: int, *, rows: int | None = None) -> None:
        """Report the first completed unit and then at the wall-clock interval."""
        self.completed = completed
        if rows is not None:
            self.rows = rows
        now = perf_counter()
        if ((not self.reported_work and completed > self.initial)
                or now - self.last_report >= self.interval):
            self._report(now)
            self.reported_work = True

    def _report(self, now: float, *, finished: bool = False) -> None:
        elapsed = max(now - self.started, 1e-9)
        rate = (self.completed - self.initial) / elapsed
        remaining = max(self.total - self.completed, 0)
        eta = f'{remaining / rate:.1f}s' if rate > 0 else 'unavailable'
        row_rate = f', {self.rows / elapsed:.1f} rows/s' if self.unit == 'updates' else ''
        status = 'finished' if finished else 'progress'
        print(f'{self.label} {self.phase}: {status} {self.completed}/{self.total} {self.unit}, '
              f'{rate:.2f} {self.unit}/s{row_rate}, elapsed {elapsed:.1f}s, ETA {eta}.', flush=True)
        self.last_report = now

    def finish_phase(self) -> None:
        """Report the phase boundary even if less than one interval has elapsed."""
        if self.phase is not None:
            self._report(perf_counter(), finished=True)
            self.phase = None
