"""
Experiment Runner for Noisy Neighbor Detection.
"""
import time
import random
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List
from agent.collector import MetricsCollector
from agent.analyzer import NoisyNeighborDetector, LLMAnalyzer, BaselineAnalyzer
from agent.rag import ConfigKnowledgeBase
from agent.loadgen import BackgroundLoadGenerator


CONTAINERS = [
    "api-gateway", "user-service", "order-service",
    "payment-service", "batch-processor",
]
PORT_MAP = {
    "api-gateway": 5001, "user-service": 5002, "order-service": 5003,
    "payment-service": 5004, "batch-processor": 5005,
}


class ExperimentRunner:
    def __init__(self, use_llm: bool = True, concurrency: int = 8, think_time: float = 0.05):
        self.collector = MetricsCollector()
        self.kb = ConfigKnowledgeBase(knowledge_path="knowledge/container_configs.json")
        self.detector = NoisyNeighborDetector(self.collector)
        self.baseline_analyzer = BaselineAnalyzer()
        self.llm_analyzer = LLMAnalyzer(self.kb) if use_llm else None
        self.loadgen = BackgroundLoadGenerator(
            concurrency_per_container=concurrency,
            think_time=think_time,
        )
        self.results: List[Dict] = []

    def inject_load(self, container: str, load_type: str = "cpu", **kwargs):
        port = PORT_MAP[container]
        payload = {"action": "start", "type": load_type, **kwargs}
        try:
            resp = requests.post(
                f"http://localhost:{port}/inject", json=payload, timeout=5
            )
            print(f"  [INJECT] {load_type} on {container}: {resp.json()}")
        except Exception as e:
            print(f"  [INJECT] Failed for {container}: {e}")

    def stop_load(self, container: str, load_type: str = "cpu"):
        port = PORT_MAP[container]
        try:
            requests.post(
                f"http://localhost:{port}/inject",
                json={"action": "stop", "type": load_type},
                timeout=5,
            )
        except Exception:
            pass

    def stop_all_injections(self):
        for c in CONTAINERS:
            for lt in ["cpu", "memory", "io"]:
                self.stop_load(c, lt)

    def collect_metrics(self, duration: int, label: str, interval: float = 1.0):
        print(f"  [{label}] Collecting {duration}s of metrics...")
        for i in range(duration):
            self.collector.collect_once()
            self.collector.collect_latency()
            if (i + 1) % 10 == 0:
                self._print_live_metrics()
            time.sleep(interval)

    def _print_live_metrics(self):
        current = self.collector.get_all_current()
        print("  [LIVE]", end="")
        for name in CONTAINERS:
            m = current.get(name, {})
            cpu = m.get("cpu_percent", 0)
            lat_df = self.collector.latency_store.get(name, pd.DataFrame())
            lat = (
                lat_df["latency_ms"].iloc[-1]
                if not lat_df.empty and "latency_ms" in lat_df.columns
                else 0
            )
            short = name.split("-")[0][:4]
            print(f"  {short}:c={cpu:.0f}%/l={lat:.0f}ms", end="")
        print()

    def run_single_trial(
        self,
        trial_num: int,
        noisy_neighbor: str = None,
        load_type: str = "cpu",
        baseline_duration: int = 15,
        load_duration: int = 30,
    ) -> Dict:
        if noisy_neighbor is None:
            noisy_neighbor = random.choice(CONTAINERS)

        victims_gt = [c for c in CONTAINERS if c != noisy_neighbor]

        print(f"\n{'='*60}")
        print(f"TRIAL {trial_num}: Noisy neighbor = {noisy_neighbor} ({load_type})")
        print(f"{'='*60}")

        # Reset metrics stores for clean baseline/contention comparison
        for name in CONTAINERS:
            self.collector.metrics_store[name] = pd.DataFrame()
            self.collector.latency_store[name] = pd.DataFrame()

        # Phase 1: Baseline (background load running, no injection)
        self.stop_all_injections()
        time.sleep(2)
        self.collect_metrics(baseline_duration, "BASELINE")

        # Phase 2: Inject noisy neighbor
        if load_type == "cpu":
            self.inject_load(noisy_neighbor, "cpu", threads=6)
        elif load_type == "memory":
            self.inject_load(noisy_neighbor, "memory", size_mb=200)
            self.inject_load(noisy_neighbor, "cpu", threads=4)
        elif load_type == "io":
            self.inject_load(noisy_neighbor, "io")
            self.inject_load(noisy_neighbor, "cpu", threads=4)

        # Phase 3: Collect under contention
        self.collect_metrics(load_duration, "CONTENTION")

        # Phase 4: Analyze
        summaries = self.collector.get_summary_stats(window=load_duration)

        # System A: threshold alerts
        baseline_alerts = self.baseline_analyzer.detect(summaries)
        system_a = self._evaluate_system_a(baseline_alerts, noisy_neighbor, victims_gt)

        # System C: delta-based cross-container detection
        detections = self.detector.detect(
            baseline_window=baseline_duration,
            contention_window=load_duration,
        )
        system_c = self._evaluate_system_c(detections, noisy_neighbor, victims_gt)

        # Print deltas
        if detections and detections[0].get("deltas_summary"):
            print("\n  [DELTAS] Baseline → Contention:")
            for name, d in detections[0]["deltas_summary"].items():
                marker = " ← SUSPECT" if name == system_c.get("top_suspect") else ""
                marker = " ← GROUND TRUTH" if name == noisy_neighbor and not marker else marker
                print(
                    f"    {name:<20} CPU: {d['baseline_cpu']:>5.1f}% → {d['contention_cpu']:>5.1f}% "
                    f"(Δ{d['cpu_delta']:>+6.1f}%) | "
                    f"Lat: {d['baseline_latency']:>6.0f}ms → {d['contention_latency']:>6.0f}ms "
                    f"(Δ{d['lat_delta_pct']:>+5.0f}%){marker}"
                )

        # LLM diagnosis
        llm_diagnosis = None
        if self.llm_analyzer and detections:
            try:
                llm_diagnosis = self.llm_analyzer.diagnose(detections[0], summaries)
            except Exception as e:
                llm_diagnosis = {"error": str(e)}

        # Phase 5: Cleanup
        self.stop_all_injections()
        self.loadgen.print_stats()

        trial_result = {
            "trial": trial_num,
            "ground_truth_noisy_neighbor": noisy_neighbor,
            "load_type": load_type,
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": summaries,
            "system_a": system_a,
            "system_c": system_c,
            "llm_diagnosis": llm_diagnosis,
            "raw_detections": detections[:3] if detections else [],
        }

        self.results.append(trial_result)
        self._print_trial_summary(trial_result)
        return trial_result

    def _evaluate_system_a(
        self, alerts: List[Dict], true_neighbor: str, victims: List[str]
    ) -> Dict:
        detected_any = len(alerts) > 0
        correctly_identified = any(a["container"] == true_neighbor for a in alerts)
        blamed_victim = any(
            a["container"] in victims and a.get("alert") == "HIGH_LATENCY"
            for a in alerts
        )
        return {
            "detected_anomaly": detected_any,
            "correctly_identified_neighbor": correctly_identified,
            "blamed_victim": blamed_victim,
            "alerts": alerts,
            "attribution_correct": correctly_identified and not blamed_victim,
        }

    def _evaluate_system_c(
        self, detections: List[Dict], true_neighbor: str, victims: List[str]
    ) -> Dict:
        if not detections:
            return {
                "detected_anomaly": False,
                "correctly_identified_neighbor": False,
                "blamed_victim": False,
                "top_suspect": None,
                "confidence": 0,
                "attribution_correct": False,
            }

        top = detections[0]
        top_suspect = top["suspect"]["name"]

        return {
            "detected_anomaly": True,
            "correctly_identified_neighbor": top_suspect == true_neighbor,
            "blamed_victim": top_suspect in victims and top_suspect != true_neighbor,
            "top_suspect": top_suspect,
            "top_victim": top["victim"]["name"],
            "confidence": top["confidence"],
            "temporal_correlation": top["temporal_correlation"],
            "suspect_cpu_delta": top["suspect"]["cpu_delta"],
            "attribution_correct": top_suspect == true_neighbor,
        }

    def _print_trial_summary(self, result: Dict):
        gt = result["ground_truth_noisy_neighbor"]
        a = result["system_a"]
        c = result["system_c"]

        print(f"\n--- Trial {result['trial']} Results ---")
        print(f"Ground truth: {gt}")
        print(
            f"System A (threshold): Correct={a['attribution_correct']}, "
            f"Blamed victim={a['blamed_victim']}"
        )
        if a["alerts"]:
            for alert in a["alerts"][:4]:  # Show max 4
                print(f"  → {alert['container']}: {alert['alert']} ({alert.get('value', 0):.1f})")
            if len(a["alerts"]) > 4:
                print(f"  ... and {len(a['alerts']) - 4} more alerts")

        print(
            f"System C (ours):      Correct={c['attribution_correct']}, "
            f"Suspect={c.get('top_suspect')}, "
            f"CPU_delta={c.get('suspect_cpu_delta', 0)}, "
            f"Confidence={c.get('confidence', 0)}"
        )

        if result.get("llm_diagnosis") and not result["llm_diagnosis"].get("error"):
            diag = result["llm_diagnosis"]
            print(f"\nLLM: {diag.get('diagnosis', '')[:400]}")

    def run_experiment(
        self,
        n_trials: int = 10,
        load_types: List[str] = None,
        baseline_duration: int = 15,
        load_duration: int = 30,
    ) -> pd.DataFrame:
        if load_types is None:
            load_types = ["cpu", "cpu", "cpu", "memory", "io"]

        print(f"\n{'#'*60}")
        print(f"STARTING EXPERIMENT: {n_trials} trials")
        print(f"{'#'*60}")

        self.loadgen.start()
        self.loadgen.wait_for_warmup(target_seconds=8)

        try:
            for i in range(n_trials):
                load_type = random.choice(load_types)
                self.run_single_trial(
                    trial_num=i + 1,
                    load_type=load_type,
                    baseline_duration=baseline_duration,
                    load_duration=load_duration,
                )
                print("\n  [COOLDOWN] 10s...")
                time.sleep(10)
        finally:
            self.loadgen.stop()

        return self.generate_report()

    def generate_report(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                "trial": r["trial"],
                "ground_truth": r["ground_truth_noisy_neighbor"],
                "load_type": r["load_type"],
                "sys_a_detected": r["system_a"]["detected_anomaly"],
                "sys_a_correct": r["system_a"]["attribution_correct"],
                "sys_a_blamed_victim": r["system_a"]["blamed_victim"],
                "sys_c_detected": r["system_c"]["detected_anomaly"],
                "sys_c_correct": r["system_c"]["attribution_correct"],
                "sys_c_suspect": r["system_c"].get("top_suspect"),
                "sys_c_confidence": r["system_c"].get("confidence", 0),
                "sys_c_cpu_delta": r["system_c"].get("suspect_cpu_delta", 0),
            })

        df = pd.DataFrame(rows)

        print(f"\n{'#'*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'#'*60}")
        print(f"Total trials: {len(df)}")
        print(f"\nSystem A (Threshold Monitoring):")
        print(f"  Detection rate:      {df['sys_a_detected'].mean():.1%}")
        print(f"  Correct attribution: {df['sys_a_correct'].mean():.1%}")
        print(f"  Blamed victim rate:  {df['sys_a_blamed_victim'].mean():.1%}")
        print(f"\nSystem C (Cross-Container Delta + RAG):")
        print(f"  Detection rate:      {df['sys_c_detected'].mean():.1%}")
        print(f"  Correct attribution: {df['sys_c_correct'].mean():.1%}")
        print(f"  Mean confidence:     {df['sys_c_confidence'].mean():.1f}")

        df.to_csv("experiment_results.csv", index=False)
        with open("experiment_results_full.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nSaved: experiment_results.csv, experiment_results_full.json")
        return df
