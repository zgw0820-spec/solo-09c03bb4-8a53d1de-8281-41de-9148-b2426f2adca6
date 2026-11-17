"""Lease ownership in the shared database.

Exactly one worker *incarnation* may hold a job's lease.  A worker's stable
role name (``worker-1`` in Compose) is not an identity: every process start
also generates a random ``instance_id``.  All authorization checks use the
triple ``(worker_id, instance_id, epoch)``.

The ``workers`` row stores the currently registered instance for each role.
Registering a new instance under an existing role is the retirement signal for
the previous instance: its heartbeats match no row, ``renew`` and the commit
fence in ``owner_ok`` fail, and its ``release`` cannot delete the successor's
lease.  A same-name successor also supersedes the old lease *immediately* with
a fresh, never-reused epoch - rolling replacement must not wait for the old
heartbeat to expire.  A worker with a **different** name may only take over
after the holder's heartbeat has been stale for ``lease_seconds``.

The epoch is the fence for checkpoint commits: ``advance_job`` verifies it
inside the same ``BEGIN IMMEDIATE`` transaction that writes the checkpoint.  A
worker whose lease was stolen therefore cannot write anything, even if its
process is alive and still believes it owns the job - it cannot roll the
breakpoint back, and it cannot report a false completion.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .db import connect, transaction


@dataclass(frozen=True)
class Lease:
    job_id: str
    worker_id: str
    epoch: int
    # Defaults to the role name for callers that predate incarnation tokens or
    # that use the lease API without registering a process.
    instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id:
            object.__setattr__(self, "instance_id", self.worker_id)


class LeaseLost(Exception):
    """This worker is no longer allowed to train the job."""


def _instance(worker_id: str, instance_id: str | None) -> str:
    return instance_id if instance_id else worker_id


def register_worker(path: Path | str, worker_id: str, instance_id: str | None = None) -> None:
    """Register this process incarnation as the role's current owner.

    An older process with the same ``worker_id`` can no longer heartbeat,
    renew, commit or release anything: every check is scoped to its own
    ``instance_id``.
    """
    instance = _instance(worker_id, instance_id)
    now = time.time()
    with transaction(path) as conn:
        conn.execute(
            """
            INSERT INTO workers(worker_id, instance_id, status, started_at, heartbeat_at)
            VALUES(?, ?, 'starting', ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                instance_id=excluded.instance_id,
                status='starting', started_at=excluded.started_at,
                heartbeat_at=excluded.heartbeat_at
            """,
            (worker_id, instance, now, now),
        )


def heartbeat(
    path: Path | str,
    worker_id: str,
    status: str,
    instance_id: str | None = None,
) -> bool:
    """Publish liveness for this incarnation.

    Returns ``False`` when a newer incarnation has registered the same role.
    Such a caller is retired and must exit without touching shared state.
    """
    instance = _instance(worker_id, instance_id)
    with transaction(path) as conn:
        updated = conn.execute(
            "UPDATE workers SET heartbeat_at=?, status=? "
            "WHERE worker_id=? AND instance_id=?",
            (time.time(), status, worker_id, instance),
        )
        return updated.rowcount == 1


def _allocate_epoch(conn: sqlite3.Connection, job_id: str) -> int:
    """Bump the job's monotonic lease counter and return the new epoch."""
    conn.execute(
        """
        INSERT INTO lease_epochs(job_id, epoch) VALUES(?, 1)
        ON CONFLICT(job_id) DO UPDATE SET epoch=lease_epochs.epoch + 1
        """,
        (job_id,),
    )
    row = conn.execute("SELECT epoch FROM lease_epochs WHERE job_id=?", (job_id,)).fetchone()
    return int(row["epoch"])


def _worker_retired(
    conn: sqlite3.Connection, worker_id: str, instance_id: str
) -> bool:
    """True when this role is registered to a different process incarnation.

    A missing ``workers`` row is tolerated for the low-level lease API; real
    workers always register before acquiring.
    """
    row = conn.execute(
        "SELECT instance_id FROM workers WHERE worker_id=?", (worker_id,)
    ).fetchone()
    return row is not None and row["instance_id"] != instance_id


def acquire(
    path: Path | str,
    job_id: str,
    worker_id: str,
    lease_seconds: float,
    instance_id: str | None = None,
) -> Lease | None:
    """Try to own ``job_id`` for this process incarnation.

    Returns a lease when the job is unowned, when the same incarnation already
    owns it (same epoch, heartbeat renewed), or when this incarnation replaces
    a previous process with the same role name (fresh epoch, immediate
    supersession).  A differently named worker may take over only after the
    holder's heartbeat is stale.  ``None`` means another live worker owns the
    job, or this incarnation has been retired by a newer one.
    """
    instance = _instance(worker_id, instance_id)
    now = time.time()
    with transaction(path) as conn:
        if _worker_retired(conn, worker_id, instance):
            return None
        row = conn.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            epoch = _allocate_epoch(conn, job_id)
            conn.execute(
                "INSERT INTO leases(job_id, worker_id, instance_id, epoch, heartbeat_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (job_id, worker_id, instance, epoch, now),
            )
            return Lease(job_id, worker_id, epoch, instance)
        if row["worker_id"] == worker_id:
            if row["instance_id"] == instance:
                conn.execute(
                    "UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, job_id)
                )
                return Lease(job_id, worker_id, int(row["epoch"]), instance)
            # Same role name, different process: this is a rolling
            # replacement.  Fence the old incarnation now, even if its
            # heartbeat is still fresh.
            epoch = _allocate_epoch(conn, job_id)
            conn.execute(
                "UPDATE leases SET worker_id=?, instance_id=?, epoch=?, heartbeat_at=? "
                "WHERE job_id=?",
                (worker_id, instance, epoch, now, job_id),
            )
            return Lease(job_id, worker_id, epoch, instance)
        if now - float(row["heartbeat_at"]) < lease_seconds:
            return None
        epoch = _allocate_epoch(conn, job_id)
        conn.execute(
            "UPDATE leases SET worker_id=?, instance_id=?, epoch=?, heartbeat_at=? "
            "WHERE job_id=?",
            (worker_id, instance, epoch, now, job_id),
        )
        return Lease(job_id, worker_id, epoch, instance)


def renew(path: Path | str, lease: Lease, lease_seconds: float) -> bool:  # noqa: ARG001
    """Renew if this exact incarnation still owns the lease.

    ``False`` means the lease moved to another owner or this process has been
    retired by a same-name successor.  The worker treats either as a hard stop.
    """
    now = time.time()
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT worker_id, instance_id, epoch FROM leases WHERE job_id=?",
            (lease.job_id,),
        ).fetchone()
        if (
            row is None
            or row["worker_id"] != lease.worker_id
            or row["instance_id"] != lease.instance_id
            or int(row["epoch"]) != lease.epoch
        ):
            return False
        if _worker_retired(conn, lease.worker_id, lease.instance_id):
            return False
        conn.execute("UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, lease.job_id))
        return True


def owner_ok(conn: sqlite3.Connection, lease: Lease) -> bool:
    """Check ownership inside an already-open write transaction.

    Both the lease row and the current role registration must still name this
    incarnation.  This closes the window between a successor registering and
    the successor receiving its lease row.
    """
    row = conn.execute(
        "SELECT worker_id, instance_id, epoch FROM leases WHERE job_id=?",
        (lease.job_id,),
    ).fetchone()
    if (
        not row
        or row["worker_id"] != lease.worker_id
        or row["instance_id"] != lease.instance_id
        or int(row["epoch"]) != lease.epoch
    ):
        return False
    return not _worker_retired(conn, lease.worker_id, lease.instance_id)


def release(path: Path | str, lease: Lease) -> None:
    """Give up a lease, but only if this exact incarnation still holds it.

    The ``DELETE`` matches ``worker_id``, ``instance_id`` and ``epoch``
    together.  A stale or retired caller's row is already gone or has a newer
    epoch, so it matches nothing: an old process can never delete its
    successor's lease, including from its shutdown handler.
    """
    with transaction(path) as conn:
        conn.execute(
            "DELETE FROM leases WHERE job_id=? AND worker_id=? AND instance_id=? AND epoch=?",
            (lease.job_id, lease.worker_id, lease.instance_id, lease.epoch),
        )


def holder(path: Path | str, job_id: str) -> tuple[str, int, float] | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch, heartbeat_at FROM leases WHERE job_id=?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return (row["worker_id"], int(row["epoch"]), float(row["heartbeat_at"]))
