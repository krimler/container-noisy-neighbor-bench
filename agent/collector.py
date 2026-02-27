"""
Metrics collector using Docker SDK.
Collects real container stats: CPU, memory, network, block IO.
Stores time-series in pandas DataFrames (lightweight alternative to Lucene for POC).
"""
import time
import docker
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional


class MetricsCollector:
    CONTAINER_NAMES = [
        "api-gateway",
        "user-service",
        "order-service",
        "payment-service",
        "batch-processor",
    ]

    PORT_MAP = {
        "api-gateway": 5001,
        "user-service": 5002,
        "order-service": 5003,
        "payment-service": 5004,
        "batch-processor": 5005,
    }

    def __init__(self, max_history: int = 600):
        """max_history: max data points per container (at 1s interval = 10 min)"""
        self.client = docker.from_env()
        self.max_history = max_history
        # Time-series store: {container_name: DataFrame}
        self.metrics_store: Dict[str, pd.DataFrame] = {
            name: pd.DataFrame() for name in self.CONTAINER_NAMES
        }
        # Application-level latency store
        self.latency_store: Dict[str, pd.DataFrame] = {
            name: pd.DataFrame() for name in self.CONTAINER_NAMES
        }

    def _calc_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            num_cpus = stats["cpu_stats"].get("online_cpus", 1)
            if system_delta > 0 and cpu_delta >= 0:
                return round((cpu_delta / system_delta) * num_cpus * 100, 2)
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0

    def _calc_memory(self, stats: dict) -> dict:
        """Extract memory usage from Docker stats."""
        try:
            mem = stats["memory_stats"]
            usage = mem.get("usage", 0)
            limit = mem.get("limit", 1)
            return {
                "memory_usage_mb": round(usage / (1024 * 1024), 2),
                "memory_limit_mb": round(limit / (1024 * 1024), 2),
                "memory_percent": round((usage / limit) * 100, 2) if limit > 0 else 0,
            }
        except (KeyError, ZeroDivisionError):
            return {"memory_usage_mb": 0, "memory_limit_mb": 0, "memory_percent": 0}

    def _calc_network(self, stats: dict) -> dict:
        """Extract network IO from Docker stats."""
        try:
            networks = stats.get("networks", {})
            rx_bytes = sum(v.get("rx_bytes", 0) for v in networks.values())
            tx_bytes = sum(v.get("tx_bytes", 0) for v in networks.values())
            return {
                "net_rx_mb": round(rx_bytes / (1024 * 1024), 4),
                "net_tx_mb": round(tx_bytes / (1024 * 1024), 4),
            }
        except (KeyError, TypeError):
            return {"net_rx_mb": 0, "net_tx_mb": 0}

    def _calc_blkio(self, stats: dict) -> dict:
        """Extract block IO from Docker stats."""
        try:
            bio = stats.get("blkio_stats", {})
            read_bytes = 0
            write_bytes = 0
            for entry in bio.get("io_service_bytes_recursive", []) or []:
                if entry.get("op") == "read":
                    read_bytes += entry.get("value", 0)
                elif entry.get("op") == "write":
                    write_bytes += entry.get("value", 0)
            return {
                "blkio_read_mb": round(read_bytes / (1024 * 1024), 4),
                "blkio_write_mb": round(write_bytes / (1024 * 1024), 4),
            }
        except (KeyError, TypeError):
            return {"blkio_read_mb": 0, "blkio_write_mb": 0}

    def collect_once(self) -> Dict[str, dict]:
        """Collect a single snapshot of metrics from all containers."""
        snapshot = {}
        ts = datetime.now()

        for name in self.CONTAINER_NAMES:
            try:
                container = self.client.containers.get(name)
                stats = container.stats(stream=False)

                metrics = {
                    "timestamp": ts,
                    "cpu_percent": self._calc_cpu_percent(stats),
                    **self._calc_memory(stats),
                    **self._calc_network(stats),
                    **self._calc_blkio(stats),
                }

                # Append to store
                new_row = pd.DataFrame([metrics])
                self.metrics_store[name] = pd.concat(
                    [self.metrics_store[name], new_row], ignore_index=True
                ).tail(self.max_history)

                snapshot[name] = metrics

            except Exception as e:
                snapshot[name] = {"error": str(e), "timestamp": ts}

        return snapshot

    def collect_latency(self) -> Dict[str, Optional[float]]:
        """Probe each container's /work endpoint and record latency."""
        latencies = {}
        ts = datetime.now()

        for name, port in self.PORT_MAP.items():
            try:
                resp = requests.get(f"http://localhost:{port}/work", timeout=5)
                data = resp.json()
                latency = data.get("latency_ms", None)
                latencies[name] = latency

                row = pd.DataFrame([{"timestamp": ts, "latency_ms": latency}])
                self.latency_store[name] = pd.concat(
                    [self.latency_store[name], row], ignore_index=True
                ).tail(self.max_history)

            except Exception:
                latencies[name] = None

        return latencies

    def get_recent_metrics(self, container: str, window: int = 30) -> pd.DataFrame:
        """Get the last `window` data points for a container."""
        df = self.metrics_store.get(container, pd.DataFrame())
        return df.tail(window)

    def get_recent_latency(self, container: str, window: int = 30) -> pd.DataFrame:
        """Get the last `window` latency measurements."""
        df = self.latency_store.get(container, pd.DataFrame())
        return df.tail(window)

    def get_all_current(self) -> Dict[str, dict]:
        """Get the most recent metric for each container."""
        current = {}
        for name in self.CONTAINER_NAMES:
            df = self.metrics_store.get(name, pd.DataFrame())
            if not df.empty:
                current[name] = df.iloc[-1].to_dict()
            else:
                current[name] = {}
        return current

    def get_summary_stats(self, window: int = 30) -> Dict[str, dict]:
        """Get mean/max/std of key metrics over the window for each container."""
        summaries = {}
        for name in self.CONTAINER_NAMES:
            df = self.metrics_store.get(name, pd.DataFrame()).tail(window)
            lat_df = self.latency_store.get(name, pd.DataFrame()).tail(window)

            if df.empty:
                summaries[name] = {}
                continue

            s = {
                "cpu_mean": round(df["cpu_percent"].mean(), 2),
                "cpu_max": round(df["cpu_percent"].max(), 2),
                "cpu_std": round(df["cpu_percent"].std(), 2),
                "mem_mean": round(df["memory_percent"].mean(), 2),
                "mem_max": round(df["memory_percent"].max(), 2),
            }

            if not lat_df.empty and "latency_ms" in lat_df.columns:
                valid = lat_df["latency_ms"].dropna()
                if not valid.empty:
                    s["latency_mean"] = round(valid.mean(), 2)
                    s["latency_p95"] = round(valid.quantile(0.95), 2)
                    s["latency_max"] = round(valid.max(), 2)

            summaries[name] = s

        return summaries
