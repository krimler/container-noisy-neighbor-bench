#!/usr/bin/env python3
"""
Batch experiment: 30 trials, 6 per container, structured and reproducible.
"""
import sys
import time
import json
import pandas as pd
from agent.collector import MetricsCollector
from agent.analyzer import NoisyNeighborDetector, LLMAnalyzer, BaselineAnalyzer
from agent.rag import ConfigKnowledgeBase
from agent.loadgen import BackgroundLoadGenerator
from agent.experiment import ExperimentRunner


def check_containers():
    import docker
    client = docker.from_env()
    expected = [
        "api-gateway", "user-service", "order-service",
        "payment-service", "batch-processor",
    ]
    for name in expected:
        try:
            c = client.containers.get(name)
            if c.status != "running":
                print(f"ERROR: {name} not running")
                return False
        except Exception:
            print(f"ERROR: {name} not found")
            return False
    print("All 5 containers running.")
    return True


def main():
    if not check_containers():
        sys.exit(1)

    # Structured trial plan: 6 trials per container, mixed load types
    CONTAINERS = [
        "api-gateway", "user-service", "order-service",
        "payment-service", "batch-processor",
    ]
    LOAD_TYPES = ["cpu", "cpu", "cpu", "cpu", "memory", "io"]

    trials = []
    trial_num = 0
    for container in CONTAINERS:
        for load_type in LOAD_TYPES:
            trial_num += 1
            trials.append({
                "trial": trial_num,
                "neighbor": container,
                "load_type": load_type,
            })

    print(f"\n{'#'*60}")
    print(f"BATCH EXPERIMENT: {len(trials)} trials")
    print(f"6 per container × 5 containers = 30 trials")
    print(f"Estimated time: ~20 minutes")
    print(f"{'#'*60}\n")

    runner = ExperimentRunner(use_llm=False, concurrency=8, think_time=0.05)

    # Start background load
    runner.loadgen.start()
    runner.loadgen.wait_for_warmup(target_seconds=8)

    try:
        for t in trials:
            runner.run_single_trial(
                trial_num=t["trial"],
                noisy_neighbor=t["neighbor"],
                load_type=t["load_type"],
                baseline_duration=10,
                load_duration=20,
            )
            print(f"\n  [COOLDOWN] 8s... ({t['trial']}/{len(trials)} done)")
            time.sleep(8)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Generating report from completed trials...")
    finally:
        runner.loadgen.stop()

    # Generate report
    df = runner.generate_report()

    # Detailed breakdown by container
    print(f"\n{'#'*60}")
    print("BREAKDOWN BY NOISY NEIGHBOR")
    print(f"{'#'*60}")
    for container in CONTAINERS:
        subset = df[df["ground_truth"] == container]
        if subset.empty:
            continue
        n = len(subset)
        a_correct = subset["sys_a_correct"].sum()
        c_correct = subset["sys_c_correct"].sum()
        c_detected = subset["sys_c_detected"].sum()
        print(f"\n  {container} ({n} trials):")
        print(f"    System A correct attribution: {a_correct}/{n} ({a_correct/n:.0%})")
        print(f"    System C detected:            {c_detected}/{n} ({c_detected/n:.0%})")
        print(f"    System C correct attribution: {c_correct}/{n} ({c_correct/n:.0%})")
        if not subset["sys_c_confidence"].empty:
            print(f"    System C mean confidence:     {subset['sys_c_confidence'].mean():.1f}")

    # Breakdown by load type
    print(f"\n{'#'*60}")
    print("BREAKDOWN BY LOAD TYPE")
    print(f"{'#'*60}")
    for lt in ["cpu", "memory", "io"]:
        subset = df[df["load_type"] == lt]
        if subset.empty:
            continue
        n = len(subset)
        c_correct = subset["sys_c_correct"].sum()
        print(f"\n  {lt} ({n} trials):")
        print(f"    System C correct: {c_correct}/{n} ({c_correct/n:.0%})")

    # Save everything
    df.to_csv("batch_results.csv", index=False)
    with open("batch_results_full.json", "w") as f:
        json.dump(runner.results, f, indent=2, default=str)
    print(f"\nSaved: batch_results.csv, batch_results_full.json")


if __name__ == "__main__":
    main()
