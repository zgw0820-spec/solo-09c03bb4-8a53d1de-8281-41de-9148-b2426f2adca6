"""Process-level integration test: real API server, two real worker processes.

Flow:
1. start ``uvicorn`` against a fresh database and wait for readiness;
2. start worker A and let it commit a few batches;
3. SIGKILL worker A (an abrupt container-style death, no cleanup);
4. start worker B, which must take over after the lease expires;
5. compare the final weights with an uninterrupted run of the same input.

Also covers a "late" worker: A is suspended between batches long enough for B
to take over, then resumed.  A must not regress the checkpoint or claim
completion.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 2.0
HEARTBEAT_INTERVAL = 0.3
BATCH_DELAY = 0.25


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE_SECONDS),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT_INTERVAL),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(BATCH_DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


class Server:
    def __init__(self, db: Path, port: int) -> None:
        self.db = db
        self.port = port
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env_for(self.db),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                self.get("/health")
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("API did not start")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for(predicate, timeout: float, interval: float = 0.05, message: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message}")


def wait_completed(server: Server, job_id: str, timeout: float = 90):
    return wait_for(
        lambda: (state := server.get(f"/jobs/{job_id}"))["status"] == "completed" and state,
        timeout=timeout,
        message="job completion",
    )


def stop_worker(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    if hard:
        proc.kill()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(records: int = 40, seed: int = 21, epochs: int = 3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 4)).tolist()
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": epochs}


def replacement_dataset():
    """Input for the rolling-replacement test, sized to outlast the pauses."""
    return dataset(records=64, seed=33, epochs=4)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    """Reference run with no interruption at all."""
    db = tmp_path_factory.mktemp("reference") / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset())
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)
    return final


@pytest.fixture(scope="module")
def uninterrupted_replacement(tmp_path_factory):
    """Reference run of the same input as the replacement test."""
    db = tmp_path_factory.mktemp("reference-replacement") / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", replacement_dataset())
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker)
    return final


def test_crash_and_takeover_matches_uninterrupted(tmp_path, uninterrupted):
    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset())
        worker_a = start_worker(db, "worker-a")
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 3,
                timeout=60,
                message="first batches",
            )
        finally:
            # Abrupt death: no cleanup handler runs, heartbeat simply stops.
            stop_worker(worker_a, hard=True)
        at_crash = server.get(f"/jobs/{job['job_id']}")
        assert at_crash["status"] == "running"
        assert 3 <= at_crash["completed_steps"] < at_crash["total_steps"]

        worker_b = start_worker(db, "worker-b")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop_worker(worker_b)

        assert final["completed_steps"] == final["total_steps"]
        assert final["total_steps"] == uninterrupted["total_steps"]
        assert final["final_weights"] == uninterrupted["final_weights"]
        assert final["loss"] == uninterrupted["loss"]
        assert final["momentum_velocity"] == uninterrupted["momentum_velocity"]

        checkpoints = server.get(f"/jobs/{job['job_id']}/checkpoints")
        assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
        workers = {c["worker_id"] for c in checkpoints}
        assert "worker-a" in workers and "worker-b" in workers
        epochs = [c["lease_epoch"] for c in checkpoints]
        assert epochs == sorted(epochs) and epochs[-1] > epochs[0]

        events = server.get(f"/jobs/{job['job_id']}/events")
        starts = [e for e in events if e["event"] == "training_started"]
        assert [e["detail"]["worker_id"] for e in starts] == ["worker-a", "worker-b"]
        assert starts[1]["detail"]["resumed_from_step"] == at_crash["completed_steps"]


def test_late_worker_cannot_regress_or_false_complete(tmp_path):
    """A is suspended between batches until its lease expires; B takes over.

    A is then resumed.  It must not write, and B's completion must stand
    unchanged.

    The pause is implemented with ``TRAINING_DEMO_PAUSE_FILE`` rather than
    SIGSTOP so it is deterministic: a process frozen by the OS while inside a
    SQLite write transaction keeps the write lock, and no takeover worker could
    progress until it was killed.
    """
    db = tmp_path / "training.db"
    pause_file = tmp_path / "pause-a"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", dataset(records=24, seed=5))
        worker_a = start_worker(db, "worker-a", TRAINING_DEMO_PAUSE_FILE=str(pause_file))
        wait_for(
            lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 2,
            timeout=60,
            message="A progress",
        )
        pause_file.write_text("paused")
        # Wait until A has stopped heartbeating so its lease can expire; by
        # then it cannot be mid-batch, so this snapshot is stable.
        wait_for(
            lambda: server.get("/workers")["leases"][0]["expired"],
            timeout=30,
            message="A lease to expire",
        )
        frozen_at = server.get(f"/jobs/{job['job_id']}")

        worker_b = start_worker(db, "worker-b")
        try:
            completed = wait_completed(server, job["job_id"])
            before_resume = server.get(f"/jobs/{job['job_id']}")
            checkpoints_before = server.get(f"/jobs/{job['job_id']}/checkpoints")

            pause_file.unlink()
            # A wakes up, finds its lease taken over, and must not write.
            time.sleep(BATCH_DELAY * 4)
            stop_worker(worker_a)

            after = server.get(f"/jobs/{job['job_id']}")
            checkpoints_after = server.get(f"/jobs/{job['job_id']}/checkpoints")
            assert after["completed_steps"] == completed["completed_steps"]
            assert after["weights"] == completed["weights"]
            assert after["momentum_velocity"] == completed["momentum_velocity"]
            assert after["status"] == "completed"
            assert after["finished_at"] == completed["finished_at"]
            assert before_resume["weights"] == completed["weights"]
            assert checkpoints_after == checkpoints_before

            steps = [c["step"] for c in checkpoints_after]
            assert steps == list(range(1, frozen_at["total_steps"] + 1))
            # No duplicate or regressed checkpoints were written by A.
            assert len(steps) == len(set(steps))
            late = [c for c in checkpoints_after if c["worker_id"] == "worker-a"]
            assert max(c["step"] for c in late) <= frozen_at["completed_steps"]
        finally:
            stop_worker(worker_a)
            stop_worker(worker_b)


def test_same_name_rolling_replacement_then_other_name_failover(
    tmp_path, uninterrupted_replacement
):
    """The rolling-replacement incident, end to end with real processes.

    1. worker-1 starts training and is paused between batches (still alive,
       heartbeat stopped, lease *not* expired).
    2. A new process is started with the same --worker-id worker-1.  It must
       immediately acquire a fresh lease epoch, without waiting for expiry.
    3. The old process is resumed.  It must not commit, must not overwrite the
       new instance's worker status, and on SIGTERM must not delete the new
       instance's lease.
    4. The new worker-1 is SIGKILLed.  worker-2 must take over after lease
       expiry, resume from the last confirmed checkpoint, and produce weights
       bit-identical to the uninterrupted reference run.
    """
    db = tmp_path / "training.db"
    pause_file = tmp_path / "pause-old"
    with Server(db, free_port()) as server:
        job = server.post("/jobs", replacement_dataset())
        job_id = job["job_id"]

        old_worker = start_worker(
            db, "worker-1", TRAINING_DEMO_PAUSE_FILE=str(pause_file)
        )
        wait_for(
            lambda: server.get(f"/jobs/{job_id}")["completed_steps"] >= 2,
            timeout=60,
            message="old worker-1 progress",
        )
        pause_file.write_text("paused")
        wait_for(
            lambda: not server.get("/workers")["workers"][0]["alive"],
            timeout=30,
            message="old worker-1 heartbeat to stop",
        )
        old_lease = server.get("/workers")["leases"][0]
        # Crucially the old lease has NOT expired yet: the replacement must not
        # depend on expiry.
        assert old_lease["expired"] is False
        old_instance = old_lease["instance_id"]
        frozen_at = server.get(f"/jobs/{job_id}")

        # New container, same role name, while the old process is still alive.
        new_worker = start_worker(db, "worker-1")
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                leases = server.get("/workers")["leases"]
                if leases and leases[0]["instance_id"] != old_instance:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("new worker-1 did not acquire a new lease")
            new_lease = server.get("/workers")["leases"][0]
            assert new_lease["worker_id"] == "worker-1"
            assert new_lease["lease_epoch"] > old_lease["lease_epoch"]
            new_instance = new_lease["instance_id"]
            assert new_instance != old_instance

            # Let the new instance commit a few batches before the old wakes.
            wait_for(
                lambda: server.get(f"/jobs/{job_id}")["completed_steps"] > frozen_at["completed_steps"],
                timeout=30,
                message="new worker-1 progress",
            )
            takeover_at = server.get(f"/jobs/{job_id}")

            # Old process resumes: fence must reject it.
            pause_file.unlink()
            time.sleep(BATCH_DELAY * 6)
            after_resume = server.get(f"/jobs/{job_id}")
            assert after_resume["completed_steps"] >= takeover_at["completed_steps"]
            checkpoints = server.get(f"/jobs/{job_id}/checkpoints")
            old_steps = [
                c["step"] for c in checkpoints if c["instance_id"] == old_instance
            ]
            assert all(step <= frozen_at["completed_steps"] for step in old_steps)

            # Graceful stop of the old process: its release must not remove the
            # new lease, and its heartbeat must not mark the role stopped.
            stop_worker(old_worker)
            leases = server.get("/workers")["leases"]
            assert leases and leases[0]["instance_id"] == new_instance
            worker_rows = {

                row["worker_id"]: row for row in server.get("/workers")["workers"]
            }
            assert worker_rows["worker-1"]["instance_id"] == new_instance
            assert worker_rows["worker-1"]["status"] != "stopped"

            # Now the new instance dies too; a differently named worker takes
            # over only after expiry.
            stop_worker(new_worker, hard=True)
            survivor = start_worker(db, "worker-2")
            try:
                final = wait_completed(server, job_id)
            finally:
                stop_worker(survivor)

        finally:
            stop_worker(old_worker)
            stop_worker(new_worker)

        assert final["status"] == "completed"
        assert final["completed_steps"] == final["total_steps"]
        assert final["final_weights"] == uninterrupted_replacement["final_weights"]
        assert final["loss"] == uninterrupted_replacement["loss"]
        assert final["momentum_velocity"] == uninterrupted_replacement["momentum_velocity"]

        checkpoints = server.get(f"/jobs/{job_id}/checkpoints")
        steps = [c["step"] for c in checkpoints]
        assert steps == list(range(1, final["total_steps"] + 1))
        owners = [(c["worker_id"], c["lease_epoch"]) for c in checkpoints]
        epochs = [epoch for _, epoch in owners]
        assert epochs == sorted(epochs)
        assert epochs[-1] > epochs[0]
        assert "worker-2" in {worker for worker, _ in owners}
