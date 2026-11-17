#!/usr/bin/env python3
"""Live rolling-replacement demo against a running API.

Reproduces the on-call incident:

1. start an old ``worker-1`` process that pauses between batches (it stops
   heartbeating but does not exit);
2. start a new process with the same ``--worker-id worker-1``: it must take
   over immediately, without waiting for the lease to expire;
3. resume the old process and stop it gracefully: it must not commit, must not
   overwrite the new instance's status, and must not delete the new lease;
4. SIGKILL the new ``worker-1``; a differently named ``worker-2`` must take
   over after lease expiry and still finish with bit-identical weights.

Usage (start the API first, e.g. ``uvicorn app.main:app --port 8011``)::

    python3 scripts/replacement_demo.py --api http://127.0.0.1:8011
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 6.0
HEARTBEAT_INTERVAL = 1.5
BATCH_DELAY = 0.6


def call(base: str, method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for(predicate, timeout: float, message: str, interval: float = 0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise SystemExit(f"timed out waiting for {message}")


def sample(records: int = 64, seed: int = 3, epochs: int = 4) -> dict:
    rng = random.Random(seed)
    features = [[round(rng.gauss(0, 1), 4) for _ in range(4)] for _ in range(records)]
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": epochs}


def worker_env(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        TRAINING_DB_PATH=str(db),
        TRAINING_LEASE_SECONDS=str(LEASE_SECONDS),
        TRAINING_HEARTBEAT_INTERVAL=str(HEARTBEAT_INTERVAL),
        TRAINING_POLL_INTERVAL="0.2",
        TRAINING_BATCH_DELAY_SECONDS=str(BATCH_DELAY),
        PYTHONUNBUFFERED="1",
    )
    env.update(extra)
    return env


def start_worker(db: Path, worker_id: str, log: Path, **extra: str):
    handle = open(log, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=worker_env(db, **extra),
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def stop(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    proc.kill() if hard else proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.environ.get("API_BASE", "http://127.0.0.1:8011"))
    parser.add_argument("--db", default=os.environ.get("TRAINING_DB_PATH"))
    args = parser.parse_args()
    if not args.db:
        raise SystemExit("set TRAINING_DB_PATH (the database the API uses)")

    base = args.api.rstrip("/")
    db = Path(args.db)
    work = Path(tempfile.mkdtemp(prefix="replacement-demo-"))
    pause_file = work / "pause-old"
    old = new = survivor = None
    try:
        print("[1] uninterrupted reference run of the same input")
        reference_db = work / "reference.db"
        # Create the reference job in the same schema the workers use, then
        # train it with an uninterrupted worker and compare bit-exact output.
        from app import store
        from app.db import init_db

        init_db(reference_db)
        payload = sample()
        reference_job = store.create_job(
            reference_db, payload["features"], payload["labels"], payload["epochs"]
        )
        reference_proc = start_worker(
            reference_db, "reference", work / "reference.log"
        )
        try:
            deadline = time.time() + 180
            while time.time() < deadline:
                state = store.get_job(reference_db, reference_job["job_id"])
                if state["status"] == "completed":
                    break
                time.sleep(0.1)
        finally:
            stop(reference_proc)
        reference = store.get_job(reference_db, reference_job["job_id"])
        assert reference["status"] == "completed"
        print(f"    reference weights={reference['final_weights']}")

        print("[2] submitting the incident job")
        job = call(base, "POST", "/jobs", payload)
        job_id = job["job_id"]
        print(f"    job_id={job_id} total_steps={job['total_steps']}")

        print("[3] old worker-1 trains, then pauses between batches")
        old = start_worker(
            db, "worker-1", work / "worker-1-old.log",
            TRAINING_DEMO_PAUSE_FILE=str(pause_file),
        )
        paused_at = wait_for(
            lambda: (state := call(base, "GET", f"/jobs/{job_id}"))["completed_steps"] >= 2
            and state,
            90,
            "old worker progress",
        )
        pause_file.write_text("paused")
        # A live worker may be mid-batch when paused; bound the frozen step by
        # the first observation after its heartbeat has stopped.
        wait_for(
            lambda: not call(base, "GET", "/workers")["workers"][0]["alive"],
            30,
            "old heartbeat to stop",
        )
        paused_at = call(base, "GET", f"/jobs/{job_id}")
        # Find the old lease by its instance, not by list position.
        old_instance = next(
            worker
            for worker in call(base, "GET", "/workers")["workers"]
            if worker["worker_id"] == "worker-1"
        )["instance_id"]
        old_lease = wait_for(
            lambda: next(
                (
                    lease
                    for lease in call(base, "GET", "/workers")["leases"]
                    if lease["instance_id"] == old_instance
                ),
                None,
            ),
            30,
            "old lease snapshot",
        )
        # The lease is still *fresh*: replacement must not depend on expiry.
        assert old_lease["expired"] is False, old_lease
        print(
            f"    paused at step {paused_at['completed_steps']},"
            f" lease epoch {old_lease['lease_epoch']},"
            f" instance {old_lease['instance_id'][:8]}"
        )

        print("[4] starting a new worker-1 while the old process is alive")
        new = start_worker(db, "worker-1", work / "worker-1-new.log")
        new_lease = wait_for(
            lambda: next(
                (
                    lease
                    for lease in call(base, "GET", "/workers")["leases"]
                    if lease["instance_id"] != old_lease["instance_id"]
                ),
                None,
            ),
            30,
            "same-name takeover",
        )
        assert new_lease["worker_id"] == "worker-1"
        assert new_lease["lease_epoch"] > old_lease["lease_epoch"]
        print(
            f"    new instance {new_lease['instance_id'][:8]}"
            f" took epoch {new_lease['lease_epoch']} without waiting for expiry"
        )
        wait_for(
            lambda: call(base, "GET", f"/jobs/{job_id}")["completed_steps"]
            > paused_at["completed_steps"],
            60,
            "successor progress",
        )

        print("[5] resuming and stopping the old process")
        pause_file.unlink()
        time.sleep(BATCH_DELAY * 8)
        state = call(base, "GET", f"/jobs/{job_id}")
        checkpoints = call(base, "GET", f"/jobs/{job_id}/checkpoints")
        old_commits = [
            c for c in checkpoints if c["instance_id"] == old_lease["instance_id"]
        ]
        assert all(c["step"] <= paused_at["completed_steps"] for c in old_commits)
        stop(old)
        old = None
        leases = call(base, "GET", "/workers")["leases"]
        assert leases and leases[0]["instance_id"] == new_lease["instance_id"]
        row = next(
            worker
            for worker in call(base, "GET", "/workers")["workers"]
            if worker["worker_id"] == "worker-1"
        )
        assert row["instance_id"] == new_lease["instance_id"] and row["status"] != "stopped"
        print(
            f"    old process fenced: no new commits, successor lease intact,"
            f" role status={row['status']!r}"
        )

        print("[6] killing the successor; worker-2 takes over after expiry")
        stop(new, hard=True)
        new = None
        survivor = start_worker(db, "worker-2", work / "worker-2.log")
        final = wait_for(
            lambda: (state := call(base, "GET", f"/jobs/{job_id}"))["status"] == "completed"
            and state,
            180,
            "completion after failover",
        )
        checkpoints = call(base, "GET", f"/jobs/{job_id}/checkpoints")
        steps = [c["step"] for c in checkpoints]
        assert steps == list(range(1, final["total_steps"] + 1)), steps
        epochs = [c["lease_epoch"] for c in checkpoints]
        assert epochs == sorted(epochs) and epochs[-1] > epochs[0]
        assert final["final_weights"] == reference["final_weights"]
        assert final["loss"] == reference["loss"]
        assert final["momentum_velocity"] == reference["momentum_velocity"]
        print(
            f"    completed {final['completed_steps']}/{final['total_steps']};"
            f" owners={sorted({c['worker_id'] for c in checkpoints})};"
            f" epochs={sorted(set(epochs))}"
        )
        print("    PASS: resumed weights are bit-identical to the uninterrupted run")
        print(f"[logs] {work}")
        return 0
    finally:
        stop(old) if old is not None else None
        stop(new) if new is not None else None
        stop(survivor) if survivor is not None else None
        pause_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
