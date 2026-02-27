"""
Background Load Generator.
Sends continuous HTTP requests to all containers to create sustained CPU utilization.
This makes the noisy neighbor effect visible - without background load,
idle containers don't compete for CPU.
"""
import time
import threading
import requests
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class LoadStats:
    """Track latency stats per container."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=500))
    errors: int = 0
    total: int = 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]


class BackgroundLoadGenerator:
    """
    Generates sustained HTTP load across all containers.
    Each container gets N concurrent "users" hitting /work continuously.
    """

    CONTAINERS = {
        "api-gateway": 5001,
        "user-service": 5002,
        "order-service": 5003,
        "payment-service": 5004,
        "batch-processor": 5005,
    }

    def __init__(self, concurrency_per_container: int = 8, think_time: float = 0.05):
        """
        concurrency_per_container: number of concurrent request loops per container
        think_time: pause between requests per thread (seconds).
                    Lower = more load. 0.05 = ~20 req/s per thread.
        """
        self.concurrency = concurrency_per_container
        self.think_time = think_time
        self.running = False
        self.threads: List[threading.Thread] = []
        self.stats: Dict[str, LoadStats] = {
            name: LoadStats() for name in self.CONTAINERS
        }

    def _worker(self, container: str, port: int):
        """Single worker loop: hit /work repeatedly."""
        url = f"http://localhost:{port}/work"
        while self.running:
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()
                latency = data.get("latency_ms", 0)
                self.stats[container].latencies.append(latency)
                self.stats[container].total += 1
            except Exception:
                self.stats[container].errors += 1
                self.stats[container].total += 1

            if self.think_time > 0:
                time.sleep(self.think_time)

    def start(self):
        """Start background load on all containers."""
        if self.running:
            return

        self.running = True
        # Reset stats
        self.stats = {name: LoadStats() for name in self.CONTAINERS}

        for container, port in self.CONTAINERS.items():
            for i in range(self.concurrency):
                t = threading.Thread(
                    target=self._worker,
                    args=(container, port),
                    daemon=True,
                    name=f"loadgen-{container}-{i}",
                )
                t.start()
                self.threads.append(t)

        total_threads = len(self.CONTAINERS) * self.concurrency
        print(
            f"  [LOADGEN] Started: {self.concurrency} workers × "
            f"{len(self.CONTAINERS)} containers = {total_threads} threads"
        )

    def stop(self):
        """Stop background load."""
        self.running = False
        # Wait briefly for threads to finish current requests
        time.sleep(1)
        self.threads.clear()
        print("  [LOADGEN] Stopped")

    def get_stats(self) -> Dict[str, dict]:
        """Get current load stats per container."""
        result = {}
        for name, s in self.stats.items():
            result[name] = {
                "total_requests": s.total,
                "errors": s.errors,
                "latency_mean": round(s.mean, 2),
                "latency_p95": round(s.p95, 2),
            }
        return result

    def print_stats(self):
        """Print current load stats."""
        print("  [LOADGEN STATS]")
        for name, s in self.stats.items():
            short = name[:15].ljust(15)
            print(
                f"    {short}  reqs={s.total:>5}  "
                f"err={s.errors:>3}  "
                f"lat_mean={s.mean:>8.1f}ms  "
                f"lat_p95={s.p95:>8.1f}ms"
            )

    def wait_for_warmup(self, target_seconds: int = 5):
        """Wait until load generator has been running and producing results."""
        print(f"  [LOADGEN] Warming up for {target_seconds}s...")
        time.sleep(target_seconds)
        self.print_stats()
