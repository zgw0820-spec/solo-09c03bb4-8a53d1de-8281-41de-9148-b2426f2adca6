"""Deterministic training tests: maths, checkpoints and crash resumption."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db, transaction
from app.features import normalise
from app.lease import LeaseLost
from app.model import batch_loss, sgd_step


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.01,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


def sample(records: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def reference_train(features, labels, epochs, lr=0.1, momentum=0.9, batch_size=8):
    """Independent replay of the training loop used as the oracle."""
    design, targets = normalise(features, labels)
    weights = np.zeros(design.shape[1])
    velocity = np.zeros(design.shape[1])
    losses = []
    steps = 0
    for _ in range(epochs):
        for start in range(0, design.shape[0], batch_size):
            weights, velocity, loss = sgd_step(
                weights,
                velocity,
                design[start : start + batch_size],
                targets[start : start + batch_size],
                lr,
                momentum,
            )
            losses.append(loss)
            steps += 1
    return weights, velocity, steps, losses


def drain(settings, job_id, lease, limit=1000):
    """Advance a job to completion as ``lease``'s owner; returns steps."""
    seen = []
    while True:
        result = store.advance_job(settings.db_path, lease)
        seen.append(result["step"])
        if result["status"] == "completed":
            return seen
        assert len(seen) < limit


def test_math_matches_reference(settings):
    features, labels = sample(11)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, job["job_id"], "w", settings.lease_seconds)
    assert drain(settings, job["job_id"], lease) == list(range(1, 5))
    final = store.get_job(settings.db_path, job["job_id"])
    w, v, steps, losses = reference_train(features, labels, 2)
    assert steps == 4
    assert np.allclose(final["weights"], w, rtol=0, atol=1e-12)
    assert np.allclose(final["momentum_velocity"], v, rtol=0, atol=1e-12)
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == [1, 2, 3, 4]
    assert np.allclose([c["batch_loss"] for c in checkpoints], losses, atol=1e-12)


def test_hyperparameters_and_batching(settings):
    features, labels = sample(18)
    job = store.create_job(settings.db_path, features, labels, 3)
    assert job["batch_size"] == 8
    assert job["learning_rate"] == 0.1
    assert job["momentum"] == 0.9
    assert job["batches_per_epoch"] == math.ceil(18 / 8) == 3
    assert job["total_steps"] == 9


def test_each_batch_is_committed_with_weights_momentum_loss_progress(settings):
    features, labels = sample(10)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, job["job_id"], "w", settings.lease_seconds)
    store.advance_job(settings.db_path, lease)
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    latest = checkpoints[-1]
    current = store.get_job(settings.db_path, job["job_id"])
    assert set(latest) >= {
        "weights",
        "momentum_velocity",
        "batch_loss",
        "loss",
        "step",
        "committed_at",
        "worker_id",
        "lease_epoch",
    }
    assert latest["step"] == current["completed_steps"]
    assert latest["weights"] == current["weights"]
    assert latest["loss"] == pytest.approx(current["loss"], abs=1e-12)
    assert current["progress"] == pytest.approx(0.25)


def test_resume_from_committed_checkpoint_matches_uninterrupted(settings):
    features, labels = sample(19, seed=4)
    uninterrupted = store.create_job(settings.db_path, features, labels, 3)
    lm.register_worker(settings.db_path, "w")
    lease = lm.acquire(settings.db_path, uninterrupted["job_id"], "w", settings.lease_seconds)
    drain(settings, uninterrupted["job_id"], lease)
    expected = store.get_job(settings.db_path, uninterrupted["job_id"])

    interrupted = store.create_job(settings.db_path, features, labels, 3)
    lease = lm.acquire(settings.db_path, interrupted["job_id"], "w", settings.lease_seconds)
    for _ in range(3):
        store.advance_job(settings.db_path, lease)
    # Simulate the process dying: lease is abandoned, then expires.
    lm.release(settings.db_path, lease)
    time.sleep(settings.lease_seconds + 0.05)
    resumed_lease = lm.acquire(
        settings.db_path, interrupted["job_id"], "w2", settings.lease_seconds
    )
    assert resumed_lease is not None
    drain(settings, interrupted["job_id"], resumed_lease)
    resumed = store.get_job(settings.db_path, interrupted["job_id"])

    assert resumed["completed_steps"] == expected["completed_steps"]
    assert resumed["final_weights"] == expected["final_weights"]
    assert resumed["momentum_velocity"] == expected["momentum_velocity"]
    assert resumed["loss"] == expected["loss"]
    steps = [c["step"] for c in store.list_checkpoints(settings.db_path, interrupted["job_id"])]
    assert steps == list(range(1, expected["total_steps"] + 1))


def test_stale_worker_cannot_write_after_handover(settings):
    features, labels = sample(20, seed=9)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)
    assert lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds) is None
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None and new.epoch == old.epoch + 1
    store.advance_job(settings.db_path, new)
    before = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == before["completed_steps"]
    assert after["weights"] == before["weights"]


def test_stale_worker_cannot_report_completion(settings):
    features, labels = sample(8, seed=2)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert new is not None
    while True:
        result = store.advance_job(settings.db_path, new)
        if result["status"] == "completed":
            break
    completed = store.get_job(settings.db_path, job["job_id"])
    assert completed["status"] == "completed"
    assert completed["completed_steps"] == completed["total_steps"]
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == completed["completed_steps"]
    assert final["finished_at"] == completed["finished_at"]


def test_late_worker_cannot_roll_back_checkpoint(settings):
    features, labels = sample(16, seed=11)
    job = store.create_job(settings.db_path, features, labels, 3)
    lm.register_worker(settings.db_path, "old")
    lm.register_worker(settings.db_path, "new")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)
    time.sleep(settings.lease_seconds + 0.05)
    new = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    for _ in range(3):
        store.advance_job(settings.db_path, new)
    snapshot = store.get_job(settings.db_path, job["job_id"])
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == snapshot["completed_steps"]
    assert after["weights"] == snapshot["weights"]


def test_input_validation(settings):
    from app.features import InputError

    features, labels = sample(7)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 2)
    features, labels = sample(8)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 1)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels, 6)
    bad_labels = list(labels)
    bad_labels[0] = 2
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, bad_labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, features, labels[:-1], 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [["x", 0.2]], labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [[0.3]], labels, 2)
    with pytest.raises(InputError):
        store.create_job(settings.db_path, [[0.1, 0.2]] * 7 + [[float("nan"), 0.2]], labels, 2)


def test_single_batch_loss_is_cross_entropy_of_zero_weights():
    features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.zeros(3)
    design = np.concatenate([features, np.ones((4, 1))], axis=1)
    assert batch_loss(weights, design, labels) == pytest.approx(math.log(2))


def test_reacquired_lease_never_reuses_an_epoch(settings):
    """Releasing a lease must not let a new owner inherit an old epoch.

    Otherwise an old worker still holding an expired Lease(job, w, 1) could
    write after a *new* owner was also given epoch 1.
    """
    features, labels = sample(16, seed=6)
    job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "old")
    first = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    assert first.epoch == 1
    lm.release(settings.db_path, first)
    second = lm.acquire(settings.db_path, job["job_id"], "new", settings.lease_seconds)
    assert second is not None and second.epoch == 2
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, first)
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 0
    store.advance_job(settings.db_path, second)
    assert store.get_job(settings.db_path, job["job_id"])["completed_steps"] == 1


def test_same_name_successor_supersedes_even_with_fresh_heartbeat(settings):
    """Rolling replacement: a new process under the same role takes over now.

    The predecessor's heartbeat is still fresh, so an expiry-based takeover
    would return ``None``; identity, not elapsed time, makes this safe.  The
    successor gets a new epoch immediately and the predecessor is fenced from
    every write, renewal and release path.
    """
    features, labels = sample(24, seed=8)
    job = store.create_job(settings.db_path, features, labels, 2)
    job_id = job["job_id"]

    lm.register_worker(settings.db_path, "worker-1", "old-instance")
    old = lm.acquire(
        settings.db_path, job_id, "worker-1", settings.lease_seconds, "old-instance"
    )
    assert old is not None and old.epoch == 1
    store.advance_job(settings.db_path, old)
    before = store.get_job(settings.db_path, job_id)

    # A replacement container registers the same role name; the old process is
    # still alive and its lease heartbeat is fresh.
    lm.register_worker(settings.db_path, "worker-1", "new-instance")
    assert lm.heartbeat(settings.db_path, "worker-1", "training", "old-instance") is False
    assert lm.acquire(
        settings.db_path, job_id, "worker-1", settings.lease_seconds, "old-instance"
    ) is None
    assert lm.renew(settings.db_path, old, settings.lease_seconds) is False

    new = lm.acquire(
        settings.db_path, job_id, "worker-1", settings.lease_seconds, "new-instance"
    )
    assert new is not None and new.epoch == old.epoch + 1
    assert new.instance_id == "new-instance"

    # The retired process cannot commit, even without ever renewing again.
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job_id)
    assert after["completed_steps"] == before["completed_steps"]
    assert after["weights"] == before["weights"]

    # Its exit-time release must not delete the successor's lease, and its
    # stop heartbeat must not overwrite the successor's status.
    lm.release(settings.db_path, old)
    held = lm.holder(settings.db_path, job_id)
    assert held is not None
    assert held[0] == "worker-1" and held[1] == new.epoch
    with transaction(settings.db_path) as conn:
        row = conn.execute(
            "SELECT instance_id, status FROM workers WHERE worker_id='worker-1'"
        ).fetchone()
    assert row["instance_id"] == "new-instance"
    assert row["status"] != "stopped"

    # The successor carries the job to the same final state as an
    # uninterrupted run.
    store.advance_job(settings.db_path, new)
    while True:
        result = store.advance_job(settings.db_path, new)
        if result["status"] == "completed":
            break
    resumed = store.get_job(settings.db_path, job_id)

    reference_job = store.create_job(settings.db_path, features, labels, 2)
    lm.register_worker(settings.db_path, "reference", "reference-instance")
    reference_lease = lm.acquire(
        settings.db_path,
        reference_job["job_id"],
        "reference",
        settings.lease_seconds,
        "reference-instance",
    )
    while True:
        result = store.advance_job(settings.db_path, reference_lease)
        if result["status"] == "completed":
            break
    expected = store.get_job(settings.db_path, reference_job["job_id"])

    assert resumed["status"] == "completed"
    assert resumed["completed_steps"] == resumed["total_steps"]
    assert resumed["final_weights"] == expected["final_weights"]
    assert resumed["momentum_velocity"] == expected["momentum_velocity"]
    assert resumed["loss"] == expected["loss"]

    steps = [c["step"] for c in store.list_checkpoints(settings.db_path, job_id)]
    assert steps == list(range(1, resumed["total_steps"] + 1))
    epochs = [c["lease_epoch"] for c in store.list_checkpoints(settings.db_path, job_id)]
    assert epochs[0] == 1 and epochs[-1] == new.epoch
    assert epochs == sorted(epochs)


def test_other_name_resumes_after_failed_same_name_successor(settings):
    """If the successor dies too, a differently named worker resumes on expiry.

    The lease chain is old worker-1 (epoch 1) -> new worker-1 (epoch 2) ->
    worker-2 (epoch 3).  Only the last transition waits for lease expiry;
    worker-2 restarts from the newest confirmed checkpoint and reproduces the
    uninterrupted weights exactly.
    """
    features, labels = sample(20, seed=12)
    reference_job = store.create_job(settings.db_path, features, labels, 3)
    lm.register_worker(settings.db_path, "reference", "ref")
    ref_lease = lm.acquire(
        settings.db_path, reference_job["job_id"], "reference", settings.lease_seconds, "ref"
    )
    while True:
        result = store.advance_job(settings.db_path, ref_lease)
        if result["status"] == "completed":
            break
    expected = store.get_job(settings.db_path, reference_job["job_id"])

    job = store.create_job(settings.db_path, features, labels, 3)
    job_id = job["job_id"]
    lm.register_worker(settings.db_path, "worker-1", "old-instance")
    old = lm.acquire(
        settings.db_path, job_id, "worker-1", settings.lease_seconds, "old-instance"
    )
    for _ in range(2):
        store.advance_job(settings.db_path, old)

    lm.register_worker(settings.db_path, "worker-1", "failed-instance")
    failed = lm.acquire(
        settings.db_path, job_id, "worker-1", settings.lease_seconds, "failed-instance"
    )
    assert failed is not None and failed.epoch == old.epoch + 1
    for _ in range(2):
        store.advance_job(settings.db_path, failed)
    confirmed = store.get_job(settings.db_path, job_id)

    # The successor crashes without releasing: no heartbeat, no cleanup.
    with pytest.raises(LeaseLost):
        store.advance_job(settings.db_path, old)
    time.sleep(settings.lease_seconds + 0.05)

    lm.register_worker(settings.db_path, "worker-2", "survivor")
    survivor = lm.acquire(
        settings.db_path, job_id, "worker-2", settings.lease_seconds, "survivor"
    )
    assert survivor is not None and survivor.epoch == failed.epoch + 1
    resumed_steps = []
    while True:
        result = store.advance_job(settings.db_path, survivor)
        resumed_steps.append(result["step"])
        if result["status"] == "completed":
            break

    resumed = store.get_job(settings.db_path, job_id)
    assert resumed_steps == list(range(confirmed["completed_steps"] + 1, resumed["total_steps"] + 1))
    assert resumed["status"] == "completed"
    assert resumed["final_weights"] == expected["final_weights"]
    assert resumed["momentum_velocity"] == expected["momentum_velocity"]
    assert resumed["loss"] == expected["loss"]
    steps = [c["step"] for c in store.list_checkpoints(settings.db_path, job_id)]
    assert steps == list(range(1, resumed["total_steps"] + 1))
    epochs = [c["lease_epoch"] for c in store.list_checkpoints(settings.db_path, job_id)]
    assert epochs == sorted(epochs)
    assert epochs[2] == old.epoch + 1 and epochs[-1] == survivor.epoch
