"""Bounded CPU-only training output workers and immutable checkpoint snapshots."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import psutil
import torch


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Publish a completed file without exposing partial writes to readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON through an atomic replacement."""
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2) + '\n')


def write_history(path: Path, rows: Sequence[Mapping[str, float]], columns: Sequence[str],
                  preserved_lines: Mapping[float, str] | None = None) -> list[str]:
    """Serialize history while preserving compatible historical CSV row text."""
    import pandas as pd
    ordered = list(columns)
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(str(key))
    preserved = preserved_lines or {}
    lines = [','.join(ordered)]
    for row in rows:
        saved = preserved.get(float(row['epoch']))
        if saved is not None and saved.count(',') == len(ordered) - 1:
            lines.append(saved)
        else:
            lines.append(pd.DataFrame([{key: row.get(key, float('nan')) for key in ordered}],
                                      columns=ordered).to_csv(index=False, header=False).rstrip('\n'))
    with atomic_output(path) as temporary:
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return ordered


def write_audit(directory: Path, epoch: int, compositions: Sequence[Mapping[str, Any]],
                summary: Mapping[str, Any] | None) -> None:
    """Write every consumed training epoch's audit, including empty pixel logs."""
    directory = Path(directory)
    with atomic_output(directory / f'epoch_{epoch:04d}_batches.jsonl') as temporary:
        with temporary.open('w') as handle:
            handle.writelines(json.dumps(row) + '\n' for row in compositions)
    if summary is not None:
        atomic_json(directory / f'epoch_{epoch:04d}_summary.json', summary)


def state_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Conservatively estimate owned tensor, array, and Python payload memory."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes + sys.getsizeof(value)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(state_bytes(k, seen) + state_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(state_bytes(v, seen) for v in value)
    return size


def training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    """Collect live training state; callers must freeze it before further updates."""
    state = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict())
    if lr_scheduler is not None:
        state["lr_scheduler_state"] = lr_scheduler.state_dict()
    return state


def cpu_snapshot(value: Any) -> Any:
    """Copy nested state, completing batched CUDA transfers before returning.

    CPU tensors must also be cloned: ``cpu()`` alone aliases live optimizer state.
    Ordered state-dict metadata is retained. No model mutation may overlap this call.
    """
    memo, streams = {}, {}

    def freeze(item):
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, torch.Tensor):
            source = item.detach()
            if source.device.type == 'cuda':
                streams[source.device] = torch.cuda.current_stream(source.device)
                result = torch.empty_like(source, device='cpu', pin_memory=True)
                result.copy_(source, non_blocking=True)
            else:
                result = source.clone()
        elif isinstance(item, dict):
            result = copy.copy(item)
            memo[id(item)] = result
            result.update((key, freeze(val)) for key, val in item.items())
            if hasattr(item, '_metadata'):
                result._metadata = copy.deepcopy(item._metadata)
        elif isinstance(item, list):
            result = [freeze(val) for val in item]
        elif isinstance(item, tuple):
            result = tuple(freeze(val) for val in item)
        else:
            result = copy.deepcopy(item)
        memo[id(item)] = result
        return result

    try:
        result = freeze(value)
    finally:
        # Also finish any already-enqueued copies if allocation/copying fails.
        for stream in streams.values():
            event = torch.cuda.Event()
            event.record(stream)
            event.synchronize()
    return result


def _init_worker():
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _ready():
    return os.getpid()


def _write_job(job):
    started = perf_counter()
    if job['history'] is not None:
        write_history(*job['history'])
    for audit in job['audits']:
        write_audit(*audit)
    for destination, payload in job['checkpoints'].items():
        with atomic_output(Path(destination)) as temporary:
            torch.save(payload, temporary)
    for summary in job['summaries']:
        print(summary, flush=True)
    return dict(kind='write', seconds=perf_counter() - started,
                checkpoint_epochs={str(path): payload['epoch'] + 1
                                   for path, payload in job['checkpoints'].items()},
                history_epoch=job['history'][1][-1]['epoch'] if job['history'] else None,
                audits=len(job['audits']))


def _plot_job(job):
    started = perf_counter()
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from dann.objective_curves import plot_objective_history
    rows, destination, dpi, config, selection = job
    with atomic_output(Path(destination)) as temporary:
        plot_objective_history(pd.DataFrame(rows), temporary, dpi, config=config, selection=selection)
    return dict(kind='plot', seconds=perf_counter() - started, history_epoch=rows[-1]['epoch'])


def _empty_write():
    return dict(history=None, checkpoints={}, audits=[], summaries=[])


def _has_write(job):
    return job['history'] is not None or any(job[key] for key in ('checkpoints', 'audits', 'summaries'))


class TrainingArtifacts:
    """One ordered writer and one renderer; pending replaceable outputs coalesce.

    A dispatcher thread keeps workers fed while the main thread trains. Only CPU
    snapshots cross process boundaries. At most one job per worker is submitted;
    replaceable best/latest and plot slots remain owned by this manager until then.
    """

    def __init__(self, directory: Path, config: Mapping[str, Any]):
        started = perf_counter()
        self.directory, self.config = Path(directory), copy.deepcopy(dict(config))
        self.condition = threading.Condition()
        self.pending_write, self.pending_plot = _empty_write(), None
        self.active = {}
        self.error = None
        self.closing = self.closed = False
        self.report = dict(mode='async', jobs=[], coalesced_checkpoints=0, coalesced_plots=0)
        from dann.curve_layout import render_reservation_bytes
        self.render_reserve = render_reservation_bytes(int(config['analysis']['figure_dpi']))
        self.budget = config['performance']['host_budget_gib'] * 2**30
        self.pools = {}
        # Reserve worker imports before launch, then use actual child RSS thereafter.
        self._check_initial_budget(2 * 2**30 + self.render_reserve)
        try:
            names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')
            saved = {name: os.environ.get(name) for name in names}
            try:
                os.environ.update({name: '1' for name in names})
                for kind in ('write', 'plot'):
                    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                               initializer=_init_worker)
                    self.pools[kind] = pool
                    pool.submit(_ready).result()
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        except BaseException:
            for pool in self.pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
            raise
        self.report['startup_seconds'] = perf_counter() - started
        self.thread = threading.Thread(target=self._dispatch, name='training-artifacts', daemon=True)
        self.thread.start()

    def _rss(self):
        parent = psutil.Process()
        total = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _check_initial_budget(self, extra):
        if self._rss() + extra > self.budget:
            raise MemoryError('Training artifacts exceed performance.host_budget_gib.')

    def _raise_error(self):
        if self.error is not None:
            raise RuntimeError('Background training artifact output failed.') from self.error

    def check(self) -> None:
        """Surface worker errors cheaply at training/validation batch boundaries."""
        with self.condition:
            self._raise_error()

    def reserve(self, extra_bytes: int) -> None:
        """Wait for required outputs only if the host budget cannot fit new work."""
        with self.condition:
            while True:
                self._raise_error()
                if self.closing:
                    raise RuntimeError('Training artifact manager is closing.')
                # RSS already includes snapshots; reserve additional IPC copies and
                # the renderer's temporary canvas buffers conservatively.
                queued = state_bytes((self.pending_write, self.pending_plot,
                                      [job for _, job in self.active.values()]))
                if self._rss() + queued + self.render_reserve + extra_bytes <= self.budget:
                    return
                if not self.active and not _has_write(self.pending_write) and self.pending_plot is None:
                    raise MemoryError('Training artifact snapshot cannot fit performance.host_budget_gib.')
                self.condition.wait(.05)

    def snapshot(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    ) -> dict[str, Any]:
        """Reserve memory before freezing model, optimizer, and optional scheduler."""
        state = training_state(model, optimizer, lr_scheduler)
        self.reserve(3 * state_bytes(state))
        return cpu_snapshot(state)

    def submit_audit(self, directory: Path, epoch: int, compositions: Sequence[Mapping[str, Any]],
                     summary: Mapping[str, Any] | None) -> None:
        """Queue a non-discardable sampling audit without waiting for disk output."""
        audit = (directory, epoch, compositions, summary)
        self.reserve(2 * state_bytes(audit))
        with self.condition:
            self.pending_write['audits'].append(copy.deepcopy(audit))
            self.condition.notify_all()

    def submit_epoch(self, rows: Sequence[Mapping[str, float]], columns: Sequence[str],
                     preserved: Mapping[float, str], selection: Mapping[str, Any],
                     payload: Mapping[str, Any] | None = None, destinations: Sequence[str] = (),
                     summary: str | None = None) -> None:
        """Keep complete history but replace pending plots and superseded saves."""
        self.reserve(2 * state_bytes((rows, columns, preserved, selection)))
        history = copy.deepcopy((self.directory / 'history.csv', rows, columns, preserved))
        plot = (history[1], self.directory / 'loss_curves.png',
                int(self.config['analysis']['figure_dpi']), self.config, copy.deepcopy(selection))
        with self.condition:
            self._raise_error()
            self.pending_write['history'] = history
            for name in destinations:
                path = self.directory / name
                if path in self.pending_write['checkpoints']:
                    self.report['coalesced_checkpoints'] += 1
                self.pending_write['checkpoints'][path] = payload
            if summary is not None:
                self.pending_write['summaries'].append(summary)
            self.report['coalesced_plots'] += int(self.pending_plot is not None)
            self.pending_plot = plot
            self.condition.notify_all()

    def _notify(self, future):
        with self.condition:
            self.condition.notify_all()

    def _dispatch(self):
        try:
            with self.condition:
                while True:
                    for kind, (future, job) in list(self.active.items()):
                        if future.done():
                            self.report['jobs'].append(future.result())
                            del self.active[kind]
                            self.condition.notify_all()
                    if 'write' not in self.active and _has_write(self.pending_write):
                        job, self.pending_write = self.pending_write, _empty_write()
                        future = self.pools['write'].submit(_write_job, job)
                        self.active['write'] = (future, job)
                        future.add_done_callback(self._notify)
                    if 'plot' not in self.active and self.pending_plot is not None:
                        job, self.pending_plot = self.pending_plot, None
                        future = self.pools['plot'].submit(_plot_job, job)
                        self.active['plot'] = (future, job)
                        future.add_done_callback(self._notify)
                    if self.closing and not self.active:
                        return
                    self.condition.wait(.05)
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def close(self) -> None:
        """Drain completed-epoch outputs before returning or post-training analysis."""
        if self.closed:
            self.check()
            return
        started = perf_counter()
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        try:
            self.thread.join()
        finally:
            for pool in self.pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            self.report['final_drain_seconds'] = perf_counter() - started
        self.check()

    def flush(self) -> None:
        """Wait for submitted outputs without closing the workers."""
        with self.condition:
            while self.active or _has_write(self.pending_write) or self.pending_plot is not None:
                self._raise_error()
                self.condition.wait(.05)
            self._raise_error()
