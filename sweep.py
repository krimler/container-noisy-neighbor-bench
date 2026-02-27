#!/usr/bin/env python3
"""
Parameter Sweep: Find which configuration produces detectable cross-container signal.

Varies:
  - Isolation mode: shares (4x for batch), equal_shares, no_limits
  - Concurrency: 4, 8, 16 workers per container
  - Noisy neighbor: batch-processor (high shares) vs user-service (equal shares)

For each combo, runs 2 quick trials and measures ALL signals:
  - CPU delta (docker stats)
  - Latency delta (application-level, victim containers)
  - Throughput delta (requests/sec change in victim containers)

Total: 3 modes × 3 concurrencies × 2 neighbors × 2 trials = 36 trials
~25 minutes

Run with Docker Desktop CPUs = 2 first, then 1 if needed.
"""
import sys
import os
import time
import json
import subprocess
import pandas as pd
from datetime import datetime
from agent.collector import MetricsCollector
from agent.analyzer import NoisyNeighborDetector, BaselineAnalyzer
from agent.rag import ConfigKnowledgeBase
from agent.loadgen import BackgroundLoadGenerator

CONTAINERS = ["api-gateway", "user-service", "order-service", "payment-service", "batch-processor"]
PORT_MAP = {
    "api-gateway": 5001, "user-service": 5002, "order-service": 5003,
    "payment-service": 5004, "batch-processor": 5005,
}

# What we're sweeping
ISOLATION_MODES = [
    ("shares", "docker-compose.shares.yml"),       # batch=1024, others=256
    ("equal_shares", "docker-compose.equalshares.yml"),  # all=512
    ("no_limits", "docker-compose.nolimits.yml"),   # no cpu constraints
]
CONCURRENCIES = [4, 8, 16]
NOISY_NEIGHBORS = ["batch-processor", "user-service"]  # one high-share, one equal


def switch_compose(compose_file: str):
    """Stop current containers and start with new compose file."""
    print(f"\n  [COMPOSE] Switching to {compose_file}...")
    subprocess.run(["docker", "compose", "down"], capture_output=True)
    time.sleep(3)
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "-d", "--build"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr}")
        return False
    # Wait for containers to be ready
    print("  [COMPOSE] Waiting 10s for containers...")
    time.sleep(10)
    # Health check
    import requests
    for name, port in PORT_MAP.items():
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code != 200:
                print(f"  [WARN] {name} not healthy")
        except Exception:
            print(f"  [WARN] {name} not reachable")
            return False
    print("  [COMPOSE] All containers ready")
    return True


def inject_load(container: str, threads: int = 6):
    import requests
    port = PORT_MAP[container]
    try:
        requests.post(
            f"http://localhost:{port}/inject",
            json={"action": "start", "type": "cpu", "threads": threads},
            timeout=5,
        )
    except Exception:
        pass


def stop_all_injections():
    import requests
    for c in CONTAINERS:
        for lt in ["cpu", "memory", "io"]:
            try:
                requests.post(
                    f"http://localhost:{PORT_MAP[c]}/inject",
                    json={"action": "stop", "type": lt},
                    timeout=3,
                )
            except Exception:
                pass


def run_trial(collector, loadgen, neighbor, trial_label):
    """Run one trial, return all measured signals."""
    BASELINE_DUR = 10
    CONTENTION_DUR = 15

    # Reset metrics
    for name in CONTAINERS:
        collector.metrics_store[name] = pd.DataFrame()
        collector.latency_store[name] = pd.DataFrame()

    stop_all_injections()
    time.sleep(2)

    # Baseline
    for _ in range(BASELINE_DUR):
        collector.collect_once()
        collector.collect_latency()
        time.sleep(1)

    # Record baseline loadgen stats
    baseline_loadgen = {}
    for name in CONTAINERS:
        s = loadgen.stats[name]
        baseline_loadgen[name] = {"total": s.total, "timestamp": time.time()}

    # Inject
    inject_load(neighbor, threads=6)

    # Contention
    for _ in range(CONTENTION_DUR):
        collector.collect_once()
        collector.collect_latency()
        time.sleep(1)

    # Record contention loadgen stats
    contention_loadgen = {}
    for name in CONTAINERS:
        s = loadgen.stats[name]
        elapsed = time.time() - baseline_loadgen[name]["timestamp"]
        reqs_during = s.total - baseline_loadgen[name]["total"]
        rps = reqs_during / max(elapsed, 1)
        contention_loadgen[name] = {
            "total": s.total,
            "reqs_during_contention": reqs_during,
            "rps": round(rps, 1),
            "latency_mean": round(s.mean, 1),
            "latency_p95": round(s.p95, 1),
        }

    stop_all_injections()

    # Compute signals
    signals = {}
    for name in CONTAINERS:
        mdf = collector.metrics_store.get(name, pd.DataFrame())
        ldf = collector.latency_store.get(name, pd.DataFrame())

        if mdf.empty or ldf.empty or "latency_ms" not in ldf.columns:
            continue

        # Split baseline / contention
        b_cpu = mdf["cpu_percent"].head(BASELINE_DUR).mean()
        c_cpu = mdf["cpu_percent"].tail(CONTENTION_DUR).mean()
        b_lat = ldf["latency_ms"].head(BASELINE_DUR).dropna().median()
        c_lat = ldf["latency_ms"].tail(CONTENTION_DUR).dropna().median()

        signals[name] = {
            "cpu_baseline": round(b_cpu, 1),
            "cpu_contention": round(c_cpu, 1),
            "cpu_delta": round(c_cpu - b_cpu, 1),
            "lat_baseline": round(b_lat, 1),
            "lat_contention": round(c_lat, 1),
            "lat_delta_pct": round(((c_lat - b_lat) / max(b_lat, 1)) * 100, 1),
            "throughput_rps": contention_loadgen[name]["rps"],
        }

    # Compute throughput baseline from first half of loadgen stats
    # (approximate - loadgen is cumulative)

    return signals


def main():
    print(f"\n{'#'*60}")
    print("PARAMETER SWEEP")
    print(f"3 isolation modes × 3 concurrencies × 2 neighbors × 2 trials = 36")
    print(f"{'#'*60}\n")

    all_results = []
    trial_num = 0

    for mode_name, compose_file in ISOLATION_MODES:
        if not switch_compose(compose_file):
            print(f"  [SKIP] Failed to start {mode_name}")
            continue

        for concurrency in CONCURRENCIES:
            think_time = 0.05

            loadgen = BackgroundLoadGenerator(
                concurrency_per_container=concurrency,
                think_time=think_time,
            )
            collector = MetricsCollector()

            loadgen.start()
            loadgen.wait_for_warmup(target_seconds=8)

            for neighbor in NOISY_NEIGHBORS:
                for rep in range(2):
                    trial_num += 1
                    label = f"{mode_name}/c={concurrency}/{neighbor}/rep{rep+1}"
                    print(f"\n--- Trial {trial_num}: {label} ---")

                    signals = run_trial(collector, loadgen, neighbor, label)

                    # Print compact signal table
                    print(f"  {'Container':<20} {'CPU Δ':>7} {'Lat Δ%':>8} {'RPS':>7}")
                    for name in CONTAINERS:
                        s = signals.get(name, {})
                        marker = " ←NN" if name == neighbor else ""
                        print(
                            f"  {name:<20} {s.get('cpu_delta', 0):>+6.1f}% "
                            f"{s.get('lat_delta_pct', 0):>+7.1f}% "
                            f"{s.get('throughput_rps', 0):>6.1f}{marker}"
                        )

                    # Determine which signals are detectable
                    # The noisy neighbor should have the HIGHEST cpu_delta
                    # OR the HIGHEST lat_delta OR the LOWEST throughput
                    if signals:
                        cpu_ranked = sorted(
                            signals.items(),
                            key=lambda x: x[1].get("cpu_delta", 0),
                            reverse=True,
                        )
                        lat_ranked = sorted(
                            signals.items(),
                            key=lambda x: x[1].get("lat_delta_pct", 0),
                            reverse=True,
                        )

                        cpu_top = cpu_ranked[0][0] if cpu_ranked else None
                        lat_top = lat_ranked[0][0] if lat_ranked else None

                        # Victim signal: which NON-neighbor container has highest lat increase?
                        victim_lat_ranked = sorted(
                            [(k, v) for k, v in signals.items() if k != neighbor],
                            key=lambda x: x[1].get("lat_delta_pct", 0),
                            reverse=True,
                        )
                        victim_top = victim_lat_ranked[0] if victim_lat_ranked else None

                        nn_signals = signals.get(neighbor, {})

                        result = {
                            "trial": trial_num,
                            "mode": mode_name,
                            "concurrency": concurrency,
                            "neighbor": neighbor,
                            "rep": rep + 1,
                            # CPU delta signal
                            "nn_cpu_delta": nn_signals.get("cpu_delta", 0),
                            "top_cpu_delta_is_nn": cpu_top == neighbor,
                            # Latency delta signal (neighbor's own latency)
                            "nn_lat_delta_pct": nn_signals.get("lat_delta_pct", 0),
                            "top_lat_delta_is_nn": lat_top == neighbor,
                            # Victim signal (cross-container effect)
                            "victim_name": victim_top[0] if victim_top else None,
                            "victim_lat_delta_pct": victim_top[1].get("lat_delta_pct", 0) if victim_top else 0,
                            "any_victim_lat_increase": any(
                                v.get("lat_delta_pct", 0) > 15
                                for k, v in signals.items() if k != neighbor
                            ),
                            # Raw signals for analysis
                            "all_signals": signals,
                        }

                        det = "✓" if result["top_cpu_delta_is_nn"] else "✗"
                        lat_det = "✓" if result["top_lat_delta_is_nn"] else "✗"
                        victim_det = "✓" if result["any_victim_lat_increase"] else "✗"
                        print(
                            f"  Detection: CPU_delta={det}  "
                            f"NN_lat_delta={lat_det}  "
                            f"Victim_cross_effect={victim_det}"
                        )

                        all_results.append(result)

                    time.sleep(5)

            loadgen.stop()

    # Summary
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "all_signals"} for r in all_results])

    print(f"\n{'#'*60}")
    print("SWEEP SUMMARY")
    print(f"{'#'*60}")

    if df.empty:
        print("No results.")
        return

    # By isolation mode
    print("\n=== BY ISOLATION MODE ===")
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        print(f"\n  {mode} ({len(sub)} trials):")
        print(f"    CPU delta detects NN:     {sub['top_cpu_delta_is_nn'].mean():.0%}")
        print(f"    Lat delta detects NN:     {sub['top_lat_delta_is_nn'].mean():.0%}")
        print(f"    Cross-container effect:   {sub['any_victim_lat_increase'].mean():.0%}")
        print(f"    Mean NN CPU delta:        {sub['nn_cpu_delta'].mean():+.1f}%")
        print(f"    Mean NN lat delta:        {sub['nn_lat_delta_pct'].mean():+.1f}%")

    # By concurrency
    print("\n=== BY CONCURRENCY ===")
    for conc in sorted(df["concurrency"].unique()):
        sub = df[df["concurrency"] == conc]
        print(f"\n  concurrency={conc} ({len(sub)} trials):")
        print(f"    CPU delta detects NN:     {sub['top_cpu_delta_is_nn'].mean():.0%}")
        print(f"    Lat delta detects NN:     {sub['top_lat_delta_is_nn'].mean():.0%}")
        print(f"    Cross-container effect:   {sub['any_victim_lat_increase'].mean():.0%}")

    # By neighbor type
    print("\n=== BY NOISY NEIGHBOR ===")
    for nn in df["neighbor"].unique():
        sub = df[df["neighbor"] == nn]
        print(f"\n  {nn} ({len(sub)} trials):")
        print(f"    CPU delta detects NN:     {sub['top_cpu_delta_is_nn'].mean():.0%}")
        print(f"    Lat delta detects NN:     {sub['top_lat_delta_is_nn'].mean():.0%}")
        print(f"    Cross-container effect:   {sub['any_victim_lat_increase'].mean():.0%}")
        print(f"    Mean NN CPU delta:        {sub['nn_cpu_delta'].mean():+.1f}%")

    # Best combo
    print("\n=== BEST CONFIGURATIONS ===")
    grouped = df.groupby(["mode", "concurrency", "neighbor"]).agg({
        "top_cpu_delta_is_nn": "mean",
        "top_lat_delta_is_nn": "mean",
        "any_victim_lat_increase": "mean",
        "nn_cpu_delta": "mean",
    }).reset_index()
    grouped["total_score"] = (
        grouped["top_cpu_delta_is_nn"] * 33 +
        grouped["top_lat_delta_is_nn"] * 33 +
        grouped["any_victim_lat_increase"] * 34
    )
    best = grouped.sort_values("total_score", ascending=False).head(10)
    print(best.to_string(index=False))

    # Save
    df.to_csv("sweep_results.csv", index=False)
    with open("sweep_results_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: sweep_results.csv, sweep_results_full.json")


if __name__ == "__main__":
    main()
