"""Same-name worker replacement must fence the old process completely.

The scenario is a rolling replacement in which the previous process has *not*
exited yet and a new process is started with the same ``--worker-id``.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from app import lease as lm
from app import store
from app.db import init_db, transaction
from app.lease import LeaseLost


def make_settings(tmp_path, **overrides):
    from app.config import Settings

    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=60.0,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def sample(records: int = 16, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.normal(size=(records, 3)).tolist(), [i % 2 for i in range(records)]


def setup_job(tmp_path, records=16, epochs=2):
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    features, labels = sample(records)
    job = store.create_job(settings.db_path, features, labels, epochs)
    return settings, job, features, labels


def test_successor_takes_over_without_waiting_for_expiry(tmp_path):
    """A same-name process is a replacement, not a second claimant."""
    settings, job, _, _ = setup_job(tmp_path)
    path = settings.db_path
    lm.register_worker(path, "worker-1", "old")
    old = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "old")
    assert old is not None and old.epoch == 1
    store.advance_job(path, old)

    lm.register_worker(path, "worker-1", "new")
    # The lease has not expired at all; identity alone authorises succession.
    new = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "new")
    assert new is not None
    assert new.epoch == 2 and new.instance_id == "new"


def test_old_instance_cannot_renew_commit_or_overwrite_status(tmp_path):
    settings, job, _, _ = setup_job(tmp_path)
    path = settings.db_path
    lm.register_worker(path, "worker-1", "old")
    old = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "old")
    store.advance_job(path, old)
    before = store.get_job(path, job["job_id"])

    lm.register_worker(path, "worker-1", "new")
    new = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "new")
    assert new is not None

    # The old process still believes it is alive: all three paths must fail.
    assert lm.heartbeat(path, "worker-1", "training", "old") is False
    assert lm.renew(path, old, settings.lease_seconds) is False
    with pytest.raises(LeaseLost):
        store.advance_job(path, old)

    after = store.get_job(path, job["job_id"])
    assert after["completed_steps"] == before["completed_steps"]
    assert after["weights"] == before["weights"]
    assert after["updated_at"] == before["updated_at"]
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT instance_id, status FROM workers WHERE worker_id='worker-1'"
        ).fetchone()
    assert row["instance_id"] == "new"
    assert row["status"] != "training"
    # A retired instance with no lease must not be able to acquire one either.
    assert lm.acquire(path, job["job_id"], "worker-1", 0.0, "old") is None


def test_old_instance_exit_does_not_delete_successor_lease_or_status(tmp_path):
    settings, job, _, _ = setup_job(tmp_path)
    path = settings.db_path
    lm.register_worker(path, "worker-1", "old")
    old = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "old")
    lm.register_worker(path, "worker-1", "new")
    new = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "new")

    # Shutdown path of the old process.
    lm.release(path, old)
    lm.heartbeat(path, "worker-1", "stopped", "old")

    assert lm.holder(path, job["job_id"])[0] == "worker-1"
    assert lm.holder(path, job["job_id"])[1] == new.epoch
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT instance_id, status FROM workers WHERE worker_id='worker-1'"
        ).fetchone()
    assert row["instance_id"] == "new"
    assert row["status"] != "stopped"
    assert lm.renew(path, new, settings.lease_seconds) is True


def test_unregistered_callers_still_fenced_by_identity_and_epoch(tmp_path):
    """The low-level lease API tolerates unregistered callers by design.

    With no ``workers`` row the lease row is the only fence: a different
    instance token, worker name or epoch must all be rejected.
    """
    settings, job, _, _ = setup_job(tmp_path)
    path = settings.db_path
    old = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "old")
    assert old is not None
    assert lm.renew(path, old, settings.lease_seconds) is True
    for impostor in (
        lm.Lease(old.job_id, old.worker_id, old.epoch, "other-instance"),
        lm.Lease(old.job_id, "worker-2", old.epoch, "old-instance"),
        lm.Lease(old.job_id, old.worker_id, old.epoch + 1, "old-instance"),
    ):
        with pytest.raises(LeaseLost):
            store.advance_job(path, impostor)
    assert store.get_job(path, job["job_id"])["completed_steps"] == 0
    assert store.advance_job(path, old)["step"] == 1


def test_retired_instance_cannot_write_even_without_release(tmp_path):
    """No release at all: the registered successor alone fences the old row."""
    settings, job, _, _ = setup_job(tmp_path)
    path = settings.db_path
    lm.register_worker(path, "worker-1", "old")
    old = lm.acquire(path, job["job_id"], "worker-1", settings.lease_seconds, "old")
    lm.register_worker(path, "worker-1", "new")
    # Successor has registered but not yet acquired the job's lease.
    with pytest.raises(LeaseLost):
        store.advance_job(path, old)
    # Even the matching lease row cannot authorise the retired incarnation.
    lm.register_worker(path, "worker-1", "new")
    with transaction(path) as conn:
        assert conn.execute(
            "SELECT instance_id FROM leases WHERE job_id=?", (job["job_id"],)
        ).fetchone()["instance_id"] == "old"
    with pytest.raises(LeaseLost):
        store.advance_job(path, old)
    assert store.get_job(path, job["job_id"])["completed_steps"] == 0


def test_completed_job_reports_confirmed_state_to_late_caller(tmp_path):
    """A takeover caller that reaches a finished job gets a loss-free result."""
    settings, job, _, _ = setup_job(tmp_path, records=8)
    path = settings.db_path
    lm.register_worker(path, "a", "a1")
    lease = lm.acquire(path, job["job_id"], "a", settings.lease_seconds, "a1")
    while True:
        result = store.advance_job(path, lease)
        if result["status"] == "completed":
            break
    lm.release(path, lease)
    time.sleep(0.01)
    lm.register_worker(path, "b", "b1")
    late = lm.acquire(path, job["job_id"], "b", settings.lease_seconds, "b1")
    if late is not None:
        result = store.advance_job(path, late)
        assert result["status"] == "completed"
        assert "loss" in result
