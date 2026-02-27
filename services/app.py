"""
Lightweight web service for each container.
Serves HTTP requests with measurable latency.
Has /inject endpoint to simulate noisy neighbor behavior.
"""
import os
import time
import threading
import math
import random
import hashlib
from flask import Flask, jsonify, request

app = Flask(__name__)
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown")
BASELINE_LATENCY_MS = int(os.environ.get("BASELINE_LATENCY_MS", "10"))

# Shared state for load injection
load_state = {
    "cpu_active": False,
    "memory_active": False,
    "memory_blob": None,
    "io_active": False,
}


def cpu_burn():
    """Burns CPU in a tight loop until told to stop."""
    while load_state["cpu_active"]:
        # Heavy computation - hash operations are CPU intensive
        data = b"burn" * 1000
        for _ in range(200):
            data = hashlib.sha256(data).digest()


def io_burn():
    """Generates disk I/O pressure."""
    while load_state["io_active"]:
        data = os.urandom(1024 * 1024)  # 1MB chunks
        with open("/tmp/io_burn.dat", "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with open("/tmp/io_burn.dat", "rb") as f:
            _ = f.read()
        time.sleep(0.01)


@app.route("/health")
def health():
    return jsonify({"service": SERVICE_NAME, "status": "healthy"})


@app.route("/work")
def work():
    """Simulates a request that takes real processing time.
    Latency increases naturally when host CPU is contended."""
    start = time.perf_counter()

    # Do substantial computation so CPU contention genuinely affects us.
    # ~50k hash operations - takes ~5-15ms on idle M1, much more under contention.
    data = os.urandom(256)
    for i in range(50000):
        data = hashlib.md5(data).digest()

    # Also do some memory allocation to feel memory pressure
    temp = bytearray(64 * 1024)  # 64KB
    for i in range(0, len(temp), 64):
        temp[i] = i % 256

    elapsed_ms = (time.perf_counter() - start) * 1000

    return jsonify({
        "service": SERVICE_NAME,
        "latency_ms": round(elapsed_ms, 2),
        "timestamp": time.time(),
    })


@app.route("/inject", methods=["POST"])
def inject_load():
    """Inject CPU, memory, or IO load to simulate noisy neighbor."""
    payload = request.get_json() or {}
    action = payload.get("action", "start")
    load_type = payload.get("type", "cpu")

    if action == "start":
        if load_type == "cpu" and not load_state["cpu_active"]:
            load_state["cpu_active"] = True
            n_threads = int(payload.get("threads", 4))
            for _ in range(n_threads):
                t = threading.Thread(target=cpu_burn, daemon=True)
                t.start()

        elif load_type == "memory" and not load_state["memory_active"]:
            size_mb = int(payload.get("size_mb", 200))
            load_state["memory_active"] = True
            blob = bytearray(size_mb * 1024 * 1024)
            for i in range(0, len(blob), 4096):
                blob[i] = 0xFF
            load_state["memory_blob"] = blob

        elif load_type == "io" and not load_state["io_active"]:
            load_state["io_active"] = True
            for _ in range(2):
                t = threading.Thread(target=io_burn, daemon=True)
                t.start()

        return jsonify({"status": "started", "type": load_type})

    elif action == "stop":
        if load_type == "cpu":
            load_state["cpu_active"] = False
        elif load_type == "memory":
            load_state["memory_active"] = False
            load_state["memory_blob"] = None
        elif load_type == "io":
            load_state["io_active"] = False

        return jsonify({"status": "stopped", "type": load_type})

    return jsonify({"error": "invalid action"}), 400


@app.route("/stats")
def stats():
    """Report current load injection state."""
    return jsonify({
        "service": SERVICE_NAME,
        "injecting_cpu": load_state["cpu_active"],
        "injecting_memory": load_state["memory_active"],
        "injecting_io": load_state["io_active"],
        "memory_blob_mb": len(load_state["memory_blob"]) // (1024*1024) if load_state["memory_blob"] else 0,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
