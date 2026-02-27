#!/usr/bin/env python3
"""
Noisy Neighbor Detection POC - Main Entry Point.

Usage:
  python main.py experiment --trials 10       # Full automated experiment
  python main.py trial --neighbor batch-processor --load cpu  # Single trial
  python main.py monitor                      # Live monitoring dashboard
  python main.py demo                         # Quick demo (3 trials)
"""
import argparse
import time
import sys
import json
from agent.collector import MetricsCollector
from agent.rag import ConfigKnowledgeBase
from agent.analyzer import NoisyNeighborDetector, LLMAnalyzer, BaselineAnalyzer
from agent.experiment import ExperimentRunner


def check_containers():
    """Verify all containers are running."""
    import docker
    client = docker.from_env()
    expected = [
        "api-gateway", "user-service", "order-service",
        "payment-service", "batch-processor",
    ]
    running = []
    missing = []

    for name in expected:
        try:
            c = client.containers.get(name)
            if c.status == "running":
                running.append(name)
            else:
                missing.append(f"{name} (status: {c.status})")
        except docker.errors.NotFound:
            missing.append(f"{name} (not found)")

    if missing:
        print(f"ERROR: Missing containers: {missing}")
        print("Run: docker compose up -d --build")
        return False

    print(f"All {len(running)} containers running.")
    return True


def check_ollama():
    """Check if Ollama is available."""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if any("llama3.2" in m or "llama3.1" in m for m in models):
            print(f"Ollama OK. Models: {models}")
            return True
        else:
            print(f"Ollama running but no llama3 model. Models: {models}")
            print("Run: ollama pull llama3.2:3b")
            return False
    except Exception:
        print("Ollama not available at localhost:11434")
        return False


def cmd_experiment(args):
    if not check_containers():
        sys.exit(1)

    use_llm = False
    if not args.no_llm:
        use_llm = check_ollama()
        if not use_llm:
            print("Continuing without LLM (metrics-only analysis)")

    runner = ExperimentRunner(
        use_llm=use_llm,
        concurrency=args.concurrency,
        think_time=args.think_time,
    )
    df = runner.run_experiment(
        n_trials=args.trials,
        baseline_duration=args.baseline,
        load_duration=args.load_duration,
    )
    print("\n=== DONE ===")
    print(df.to_string())


def cmd_trial(args):
    if not check_containers():
        sys.exit(1)

    use_llm = check_ollama() if not args.no_llm else False

    runner = ExperimentRunner(
        use_llm=use_llm,
        concurrency=args.concurrency,
        think_time=args.think_time,
    )

    # Start background load
    runner.loadgen.start()
    runner.loadgen.wait_for_warmup(target_seconds=8)

    try:
        result = runner.run_single_trial(
            trial_num=1,
            noisy_neighbor=args.neighbor,
            load_type=args.load,
            baseline_duration=args.baseline,
            load_duration=args.load_duration,
        )
    finally:
        runner.loadgen.stop()

    print("\n=== FULL RESULT ===")
    print(json.dumps(result, indent=2, default=str))


def cmd_monitor(args):
    if not check_containers():
        sys.exit(1)

    collector = MetricsCollector()
    print("\nLive monitoring (Ctrl+C to stop)...\n")

    try:
        while True:
            snapshot = collector.collect_once()
            latencies = collector.collect_latency()

            print("\033[2J\033[H", end="")
            print(f"{'='*80}")
            print(f"  CONTAINER METRICS  |  {time.strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            print(f"{'Container':<20} {'CPU%':>8} {'Mem%':>8} {'MemMB':>8} {'Latency':>10}")
            print(f"{'-'*80}")

            for name in collector.CONTAINER_NAMES:
                m = snapshot.get(name, {})
                lat = latencies.get(name)
                if "error" in m:
                    print(f"{name:<20} {'ERROR':>8}")
                    continue

                lat_str = f"{lat:.1f}ms" if lat else "N/A"
                cpu = m.get("cpu_percent", 0)
                cpu_color = "\033[91m" if cpu > 50 else "\033[93m" if cpu > 30 else "\033[92m"
                reset = "\033[0m"

                print(
                    f"{name:<20} {cpu_color}{cpu:>7.1f}%{reset} "
                    f"{m.get('memory_percent', 0):>7.1f}% "
                    f"{m.get('memory_usage_mb', 0):>7.1f} "
                    f"{lat_str:>10}"
                )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def cmd_demo(args):
    """Quick demo: 3 trials with background load running."""
    print("\n=== QUICK DEMO (3 trials with background load) ===")
    print("Background load creates sustained CPU utilization.")
    print("Noisy neighbor injection on top causes contention → victim latency rises.\n")

    if not check_containers():
        sys.exit(1)

    use_llm = check_ollama() if not args.no_llm else False
    runner = ExperimentRunner(
        use_llm=use_llm,
        concurrency=args.concurrency,
        think_time=args.think_time,
    )

    # Start background load
    runner.loadgen.start()
    runner.loadgen.wait_for_warmup(target_seconds=8)

    try:
        # Trial 1: batch-processor (expected noisy neighbor)
        runner.run_single_trial(
            1, "batch-processor", "cpu", baseline_duration=10, load_duration=20
        )
        time.sleep(5)

        # Trial 2: user-service (unexpected noisy neighbor)
        runner.run_single_trial(
            2, "user-service", "cpu", baseline_duration=10, load_duration=20
        )
        time.sleep(5)

        # Trial 3: random
        runner.run_single_trial(
            3, load_type="cpu", baseline_duration=10, load_duration=20
        )
    finally:
        runner.loadgen.stop()

    df = runner.generate_report()
    print("\n=== DEMO COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(description="Noisy Neighbor Detection POC")

    # Common args
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Background load: concurrent requests per container (default: 8)"
    )
    parser.add_argument(
        "--think-time", type=float, default=0.05,
        help="Background load: pause between requests in seconds (default: 0.05)"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Experiment
    exp = subparsers.add_parser("experiment", help="Run full experiment")
    exp.add_argument("--trials", type=int, default=10)
    exp.add_argument("--baseline", type=int, default=15)
    exp.add_argument("--load-duration", type=int, default=30)
    exp.add_argument("--no-llm", action="store_true")

    # Single trial
    trial = subparsers.add_parser("trial", help="Run single trial")
    trial.add_argument("--neighbor", type=str, default=None)
    trial.add_argument("--load", type=str, default="cpu", choices=["cpu", "memory", "io"])
    trial.add_argument("--baseline", type=int, default=15)
    trial.add_argument("--load-duration", type=int, default=30)
    trial.add_argument("--no-llm", action="store_true")

    # Monitor
    subparsers.add_parser("monitor", help="Live monitoring")

    # Demo
    demo = subparsers.add_parser("demo", help="Quick 3-trial demo")
    demo.add_argument("--no-llm", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "experiment": cmd_experiment,
        "trial": cmd_trial,
        "monitor": cmd_monitor,
        "demo": cmd_demo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
