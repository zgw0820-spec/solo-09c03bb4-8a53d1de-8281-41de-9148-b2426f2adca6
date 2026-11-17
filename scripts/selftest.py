#!/usr/bin/env python3
"""Self-checks that do not need Docker.

1. ``config``: the Compose file is valid and wires two workers and the API to
   one shared volume.
2. ``equivalence``: with real worker processes, kill one mid-training and prove
   the finished weights are bit-identical to an uninterrupted run.
3. ``same-name``: a replacement process started under the same worker id takes
   over immediately, the retired process cannot write, overwrite status or
   release the successor's lease, and a later differently named worker still
   resumes to bit-identical weights.

Each check prints PASS/FAIL and exits non-zero if anything fails.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASE = 2.0
HEARTBEAT = 0.3
DELAY = 0.2


def check_compose() -> bool:
    try:
        import yaml
    except ImportError:
        print("  SKIP: PyYAML not installed")
        return True
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = config.get("services", {})
    workers = [
        name
        for name, service in services.items()
        if name.startswith("worker-") and not service.get("profiles")
    ]
    assert services["api"]["ports"] == ["8000:8000"], services["api"]["ports"]
    assert sorted(workers) == ["worker-1", "worker-2"], workers
    for name in workers:
        volumes = services[name]["volumes"]
        assert volumes == ["training-data:/data"], (name, volumes)
        assert services[name]["command"][-1] == name, services[name]["command"]
    assert services["api"]["volumes"] == ["training-data:/data"]
    assert "training-data" in config["volumes"]
    replacement = services.get("worker-1-replacement")
    if replacement is not None:
        # The replacement must run the *same* role name as worker-1.
        assert replacement["command"][-1] == "worker-1", replacement["command"]
        assert replacement.get("profiles"), "replacement must be opt-in"
    print(f"  PASS: compose has api + {workers} sharing volume training-data")
    return True


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


def get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read())


def post(port: int, path: str, body: dict):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for(predicate, timeout: float, message: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise SystemExit(f"  FAIL: timed out waiting for {message}")


def start_api(db: Path) -> tuple[subprocess.Popen, int]:
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=ROOT,
        env=env_for(db),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for(lambda: _probe(port), 30, "API")
    return proc, port


def _probe(port: int) -> bool:
    try:
        get(port, "/health")
        return True
    except OSError:
        return False


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    proc.kill() if hard else proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(seed: int = 21, records: int = 40) -> dict:
    import random

    rng = random.Random(seed)
    features = [
        [round(rng.gauss(0, 1), 4) for _ in range(4)] for _ in range(records)
    ]
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": 3}


def completed_state(port: int, job_id: str) -> dict:
    return wait_for(
        lambda: (state := get(port, f"/jobs/{job_id}"))["status"] == "completed" and state,
        120,
        "job completion",
    )


def run_reference(tmp: Path) -> dict:
    db = tmp / "reference.db"
    api, port = start_api(db)
    worker = start_worker(db, "solo")
    try:
        job = post(port, "/jobs", dataset())
        return completed_state(port, job["job_id"])
    finally:
        stop(worker)
        stop(api)


def check_equivalence(tmp: Path) -> bool:
    reference = run_reference(tmp)
    db = tmp / "crash.db"
    api, port = start_api(db)
    job = post(port, "/jobs", dataset())
    worker_a = start_worker(db, "worker-a")
    try:
        wait_for(
            lambda: get(port, f"/jobs/{job['job_id']}")["completed_steps"] >= 3,
            60,
            "first batches",
        )
    finally:
        stop(worker_a, hard=True)
    at_crash = get(port, f"/jobs/{job['job_id']}")
    worker_b = start_worker(db, "worker-b")
    try:
        final = completed_state(port, job["job_id"])
        checkpoints = get(port, f"/jobs/{job['job_id']}/checkpoints")
        events = get(port, f"/jobs/{job['job_id']}/events")
    finally:
        stop(worker_b)
        stop(api)

    assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
    assert {c["worker_id"] for c in checkpoints} == {"worker-a", "worker-b"}
    resumed = [e for e in events if e["event"] == "training_started"]
    assert resumed[-1]["detail"]["resumed_from_step"] == at_crash["completed_steps"]
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]
    assert final["final_weights"] == reference["final_weights"], (
        final["final_weights"],
        reference["final_weights"],
    )
    assert final["loss"] == reference["loss"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    print(
        f"  PASS: killed at step {at_crash['completed_steps']},"
        f" resumed to {final['completed_steps']}, weights match the uninterrupted run"
    )
    return True


def check_late_worker(tmp: Path) -> bool:
    db = tmp / "late.db"
    pause = tmp / "pause"
    api, port = start_api(db)
    job = post(port, "/jobs", dataset(seed=7, records=24))
    worker_a = start_worker(db, "worker-a", TRAINING_DEMO_PAUSE_FILE=str(pause))
    try:
        wait_for(
            lambda: get(port, f"/jobs/{job['job_id']}")["completed_steps"] >= 2,
            60,
            "A progress",
        )
        pause.write_text("pause")
        wait_for(
            lambda: get(port, "/workers")["leases"][0]["expired"],
            30,
            "A lease expiry",
        )
        frozen = get(port, f"/jobs/{job['job_id']}")
        worker_b = start_worker(db, "worker-b")
        try:
            final = completed_state(port, job["job_id"])
            before = get(port, f"/jobs/{job['job_id']}/checkpoints")
            pause.unlink()
            time.sleep(DELAY * 6)
            after = get(port, f"/jobs/{job['job_id']}")
            after_checkpoints = get(port, f"/jobs/{job['job_id']}/checkpoints")
        finally:
            stop(worker_a)
            stop(worker_b)
            stop(api)
    finally:
        stop(worker_a)
        pause.unlink(missing_ok=True)

    assert after["completed_steps"] == final["completed_steps"]
    assert after["weights"] == final["weights"]
    assert after["finished_at"] == final["finished_at"]
    assert after_checkpoints == before
    assert max(
        c["step"] for c in before if c["worker_id"] == "worker-a"
    ) <= frozen["completed_steps"]
    print(
        f"  PASS: old worker frozen at step {frozen['completed_steps']},"
        f" new worker finished at {final['completed_steps']}, no rollback or false completion"
    )
    return True


def check_same_name_replacement(tmp: Path) -> bool:
    """Rolling replacement with a reused worker id, then failover to another."""
    reference_db = tmp / "same-name-reference.db"
    api, port = start_api(reference_db)
    worker = start_worker(reference_db, "solo")
    try:
        job = post(port, "/jobs", dataset(seed=33, records=64))
        reference = completed_state(port, job["job_id"])
    finally:
        stop(worker)
        stop(api)

    db = tmp / "same-name.db"
    pause = tmp / "pause-old"
    api, port = start_api(db)
    job = post(port, "/jobs", dataset(seed=33, records=64))
    job_id = job["job_id"]
    old = start_worker(db, "worker-1", TRAINING_DEMO_PAUSE_FILE=str(pause))
    new = None
    survivor = None
    try:
        wait_for(
            lambda: get(port, f"/jobs/{job_id}")["completed_steps"] >= 2,
            60,
            "old worker-1 progress",
        )
        pause.write_text("pause")
        wait_for(
            lambda: not get(port, "/workers")["workers"][0]["alive"],
            30,
            "old worker-1 heartbeat stop",
        )
        old_lease = get(port, "/workers")["leases"][0]
        assert old_lease["expired"] is False, old_lease
        old_instance = old_lease["instance_id"]
        frozen = get(port, f"/jobs/{job_id}")

        new = start_worker(db, "worker-1")
        wait_for(
            lambda: (
                leases := get(port, "/workers")["leases"]
            )
            and leases[0]["instance_id"] != old_instance,
            30,
            "same-name takeover",
        )
        takeover = get(port, "/workers")["leases"][0]
        assert takeover["worker_id"] == "worker-1"
        assert takeover["lease_epoch"] > old_lease["lease_epoch"], takeover
        new_instance = takeover["instance_id"]
        wait_for(
            lambda: get(port, f"/jobs/{job_id}")["completed_steps"]
            > frozen["completed_steps"],
            30,
            "successor progress",
        )
        before_wake = get(port, f"/jobs/{job_id}")

        pause.unlink()
        time.sleep(DELAY * 8)
        after_wake = get(port, f"/jobs/{job_id}")
        checkpoints_now = get(port, f"/jobs/{job_id}/checkpoints")
        assert after_wake["completed_steps"] >= before_wake["completed_steps"]
        assert all(
            c["step"] <= frozen["completed_steps"]
            for c in checkpoints_now
            if c["instance_id"] == old_instance
        ), "old instance wrote after replacement"

        stop(old)
        leases = get(port, "/workers")["leases"]
        assert leases and leases[0]["instance_id"] == new_instance, leases
        workers = {w["worker_id"]: w for w in get(port, "/workers")["workers"]}
        assert workers["worker-1"]["instance_id"] == new_instance
        assert workers["worker-1"]["status"] != "stopped", workers["worker-1"]

        stop(new, hard=True)
        new = None
        survivor = start_worker(db, "worker-2")
        final = completed_state(port, job_id)
        checkpoints = get(port, f"/jobs/{job_id}/checkpoints")
        events = get(port, f"/jobs/{job_id}/events")
    finally:
        stop(old)
        if new is not None:
            stop(new)
        if survivor is not None:
            stop(survivor)
        pause.unlink(missing_ok=True)
        stop(api)

    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, final["total_steps"] + 1)), steps
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs == sorted(epochs), epochs
    assert epochs[-1] > epochs[0]
    assert {c["worker_id"] for c in checkpoints} == {"worker-1", "worker-2"}
    starts = [e for e in events if e["event"] == "training_started"]
    assert [e["detail"]["worker_id"] for e in starts].count("worker-1") >= 2
    assert starts[-1]["detail"]["worker_id"] == "worker-2"
    assert starts[-1]["detail"]["resumed_from_step"] >= frozen["completed_steps"]
    assert final["final_weights"] == reference["final_weights"], (
        final["final_weights"],
        reference["final_weights"],
    )
    assert final["loss"] == reference["loss"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    print(
        f"  PASS: old worker-1 frozen at step {frozen['completed_steps']},"
        f" same-name successor fenced it (epoch {old_lease['lease_epoch']} ->"
        f" {takeover['lease_epoch']}), worker-2 resumed to"
        f" {final['completed_steps']} with reference weights"
    )
    return True


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for name, check in (
            ("compose", check_compose),
            ("killed-worker equivalence", lambda: check_equivalence(tmp)),
            ("late-worker fencing", lambda: check_late_worker(tmp)),
            ("same-name replacement", lambda: check_same_name_replacement(tmp)),
        ):
            print(f"[{name}]")
            try:
                check()
            except AssertionError as exc:
                print(f"  FAIL: {exc}")
                failures.append(name)
        if failures:
            print(f"\nFAILED: {failures}")
            return 1
    print("\nAll self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
